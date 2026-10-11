"""A1 loss insertion, immutable buffers, and epoch/stage transactional recovery."""
import copy
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
from .bindings import bind
bind()
from .baseline_loop import A1EpochLoop
from .native_observe import NativeLossMixin
from .io import save,write,sha
from .state import capture,restore,rng_state,restore_rng,cpu,capture_device
from .sampling import NativeSampler,planned_cache
from .references import ensure_stage
from .geometry import project,angle_hinge
from libero.lifelong.algos.mlr import MLR,update_buffer_equal_quota_10,BUFFER_CAPACITY


class A1MLR(NativeLossMixin,A1EpochLoop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root=Path(self.cfg.a1_root);self.report=Path(self.cfg.a1_report)
        self.report.mkdir(parents=True,exist_ok=True)
        self.recovery=Path(self.cfg.a1_recovery)
        self.resume=None;self.a1_stage=None;self.a1_start_epoch=0
        self.a1_totals=dict(steps=0,current_windows=0,old_windows=0)
        if self.recovery.exists():
            self.resume=torch.load(self.recovery,map_location='cpu')
            assert self.resume['protocol_sha256']==sha(self.cfg.a1_protocol),'A1_PROTOCOL_MISMATCH'
            assert self.resume['implementation_files']==self.implementation_files(),'A1_CODE_MISMATCH'
            self.cfg.buffer_dir=self.resume['buffer_dir']
            assert sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt'))==self.resume['buffer_files']
            assert all(sha(self.root/'subspaces'/n)==h for n,h in self.resume['subspace_files'].items())
            self.a1_totals=self.resume['totals']
            self._retain_committed_logs()

    def implementation_files(self):
        return json.loads(Path(self.cfg.a1_protocol).read_text())['implementation_files']

    def _retain_committed_logs(self):
        for p in [Path(self.experiment_dir)/'lr_protocol.csv',Path(self.experiment_dir)/'paper_repro_train_curve.csv',self.report/'training_log.csv',Path(self.cfg.ifa_interaction_log_dir)/('seed%d_ifa_epoch_stats.csv'%self.cfg.seed)]:
            if not p.exists():continue
            with p.open() as f:r=csv.DictReader(f);keys=r.fieldnames;rows=list(r)
            keep=[r for r in rows if int(r.get('stage',r.get('task')))<self.resume['task'] or
                (int(r.get('stage',r.get('task')))==self.resume['task'] and int(r['epoch'])<self.resume['next_epoch'])]
            if len(keep)!=len(rows):
                os.replace(p,p.with_name(p.name+'.interrupted.%d'%time.time_ns()))
                with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(keep)

    def start_task(self,task,train_loader):
        super().start_task(task,train_loader)
        if self.resume:
            if self.resume['task']==self.a1_stage:restore(self,self.resume['state'])
            elif self.resume['task']<self.a1_stage and self.resume['stage_complete']:
                self.policy.load_state_dict(self.resume['state']['model'],strict=True)
                restore_rng(self.resume['state']['rng'])
            else:raise RuntimeError('Unexpected A1 recovery stage')

    def make_sampler(self,dataset,task):
        ensure_stage(self,task)
        self.active_sampler=NativeSampler(dataset,task)
        return self.active_sampler

    def project_ifa(self,z):
        if not self.cfg.a1_enabled:return z
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();out=project(z,self.a1_U);end.record()
        self._projection_events=(start,end)
        with torch.no_grad():
            zn=z.detach().norm(dim=-1);dn=out.detach().norm(dim=-1)
            ratio=dn/zn.clamp_min(1e-8)
            self._a1_diag.update(count=z.shape[0],z_norm_sum=float(zn.sum()),projected_norm_sum=float(dn.sum()),
                rho_sum=float(ratio.sum()),energy_ratio_sum=float((ratio**2).sum()),rho_min=float(ratio.min()),
                small_projection_count=int((ratio<1e-6).sum()),rank=self.a1_U.shape[1])
            self._global_z=z.detach() if (self._optimizer_step_count+1)%100==0 else None
        return out

    def record_ifa_pair(self,pos,neg,margin,angle_pos,angle_neg,hinge):
        with torch.no_grad():
            row=dict(positive_angle_mean=float(angle_pos.mean()),negative_angle_mean=float(angle_neg.mean()),
                margin=float(margin),hinge_mean=float(hinge.mean()),active_fraction=float((hinge>0).float().mean()))
            if self._global_z is not None:
                g,_,_=angle_hinge(self._global_z,pos.detach(),neg.detach(),margin)
                row['same_z_global_hinge_mean']=float(g.mean())
                # Preserve the original repeated-normalization state across pairs.
                for _ in range(2):self._global_z=self._global_z/(self._global_z.norm(dim=-1,keepdim=True)+1e-8)
            if self.cfg.a1_enabled:
                refs=torch.cat([pos.detach(),neg.detach()])
                unit=refs/refs.norm(dim=-1,keepdim=True)
                err=float((unit-project(unit,self.a1_U)).norm(dim=-1).max())
                assert err<=1e-5,('IFA_REFERENCE_OUTSIDE_FROZEN_SUBSPACE',err)
                row['actual_reference_reconstruction_max']=err
            self._a1_diag.setdefault('pairs',[]).append(row)

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        self._a1_diag={};self._projection_events=None;self._global_z=None
        model=getattr(self.policy,'module',self.policy);original=model.policy_head.loss_fn
        self.bc_per=None
        def logged_bc(gmm,target,reduction='mean'):
            loss=original(gmm,target,reduction)
            if reduction=='mean':
                with torch.no_grad():self.bc_per=original(gmm,target,'none').detach().cpu().tolist()
            return loss
        model.policy_head.loss_fn=logged_bc
        pre=None
        try:
            pre=capture_device(self)
            return NativeLossMixin.observe(self,data,current_task_id,epoch,ifa_info)
        except Exception as e:
            save(self.report/('FAILED_t%d_e%d_s%d.pt'%(current_task_id,epoch,self._optimizer_step_count)),
                dict(error=repr(e),task=current_task_id,epoch=epoch,ids=list(data[2]),batch=cpu(data),subspace=cpu(self.a1_subspace_state),pre_update_state=pre if pre is not None else getattr(self,'_failure_pre_state',None),failed_state_not_recovery=capture(self)))
            raise
        finally:model.policy_head.loss_fn=original;self.bc_per=None

    def finish_native(self,loss,data,task,epoch,bc_loss,aux_terms):
        before=(self._optimizer_step_count,self._scheduler_step_count)
        lr=float(self.optimizer.param_groups[0]['lr'])
        if not bool(torch.isfinite(loss)):
            self._failure_pre_state=capture(self)
            raise FloatingPointError('NONFINITE_LOSS')
        (self.loss_scale*loss).backward()
        gn=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
        if not bool(torch.isfinite(gn)):
            self._failure_pre_state=capture(self)
            raise FloatingPointError('NONFINITE_GRADIENT')
        self.optimizer_step_and_maybe_schedule()
        if not bool(torch.stack([torch.isfinite(p).all() for p in self.policy.parameters()]).all()):
            raise FloatingPointError('NONFINITE_PARAMETERS_AFTER_NATIVE_STEP')
        assert self._optimizer_step_count==before[0]+1 and self._scheduler_step_count==before[1]+1
        tasks=[int(t) for t in data[1]];cur=[v for v,t in zip(self.bc_per,tasks) if t==task];old=[v for v,t in zip(self.bc_per,tasks) if t<task]
        row=dict(seed=self.cfg.seed,task=task,epoch=epoch,step=self._optimizer_step_count,
            ids=list(data[2]),source_tasks=tasks,current_count=len(cur),old_count=len(old),bc=float(bc_loss.detach()),
            current_bc=sum(cur)/len(cur) if cur else None,old_bc=sum(old)/len(old) if old else None,
            weighted_ifa=[float(t.detach()) for t in aux_terms],total_loss=float(loss.detach()),lr=lr,
            gradient_norm=float(gn),ifa_stats=getattr(self,'_last_ifa_interaction_stats',None),a1=self._a1_diag)
        if self._projection_events:
            self._projection_events[1].synchronize()
            row['a1']['projection_ms']=self._projection_events[0].elapsed_time(self._projection_events[1])
        path=self.report/'events'/('task%d_epoch%03d.jsonl'%(task,epoch));path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.a1_totals['steps']+=1;self.a1_totals['current_windows']+=len(cur);self.a1_totals['old_windows']+=len(old)
        return loss.item()

    def checkpoint(self,next_epoch,stage_complete=False):
        value=dict(state=capture(self),task=self.a1_stage,next_epoch=next_epoch,stage_complete=stage_complete,
            lifecycle='BUFFER_COMMITTED' if stage_complete else ('EP99_SAVED' if next_epoch==100 else 'TRAINING'),
            subspace_state=cpu(self.a1_subspace_state),totals=self.a1_totals.copy(),buffer_dir=self.cfg.buffer_dir,
            buffer_files=sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt')),
            subspace_files={p.name:sha(p) for p in (self.root/'subspaces').glob('*.pt')},
            protocol_sha256=sha(self.cfg.a1_protocol),implementation_files=self.implementation_files())
        save(self.recovery,value)
        write(self.report/'STATUS.json',dict(status=value['lifecycle'],task=self.a1_stage,next_epoch=next_epoch,
              seed=self.cfg.seed,optimizer_step=self._optimizer_step_count,scheduler_step=self._scheduler_step_count,
              totals=self.a1_totals,recovery=str(self.recovery),a1_training_active=not stage_complete,e1_enabled=False))

    def _append_csv(self,path,fieldnames,row):
        MLR._append_csv(path,fieldnames,row)
        if Path(path).name=='paper_repro_train_curve.csv':
            events=self.report/'events'/('task%d_epoch%03d.jsonl'%(row['stage'],row['epoch']))
            rows=[json.loads(s) for s in events.read_text().splitlines()]
            sampler=self.active_sampler
            counts={str(t):sum(r['source_tasks'].count(t) for r in rows) for t in range(6,int(row['stage'])+1)}
            assert sum(r['current_count'] for r in rows)==sampler.last['current_count']
            assert sum(r['old_count'] for r in rows)==sampler.last['old_count']
            write(self.report/'sampling'/('task%d_epoch%03d.json'%(row['stage'],row['epoch'])),dict(sampler.last,actual_counts=counts))
            summary=dict(stage=row['stage'],epoch=row['epoch'],steps=len(rows),
                bc=sum(r['bc'] for r in rows)/len(rows),total_loss=sum(r['total_loss'] for r in rows)/len(rows),
                ifa=sum(sum(r['weighted_ifa']) for r in rows)/len(rows),current_windows=sum(r['current_count'] for r in rows),
                old_windows=sum(r['old_count'] for r in rows),lr_end=row['lr_end'])
            MLR._append_csv(str(self.report/'training_log.csv'),list(summary),summary)
            self.checkpoint(int(row['epoch'])+1)

    def commit_stage(self):
        task=self.a1_stage;transaction=self.root/'transactions'/('task%d'%task)
        transaction.mkdir(parents=True,exist_ok=True)
        planned=transaction/'PLAN.json';buffer=transaction/'buffer'
        # Work only in A1-owned staging storage. Completed selection is persisted
        # before any buffer work; recovery never resamples a completed plan.
        if planned.exists():
            plan=json.loads(planned.read_text());assert sorted(p.name for p in buffer.glob('*.pt'))==plan['files']
        else:
            if buffer.exists():
                os.replace(buffer,transaction/('interrupted_buffer_%d'%time.time_ns()))
            buffer.mkdir()
            for p in Path(self.cfg.buffer_dir).glob('*.pt'):os.link(p,buffer/p.name)
            update_buffer_equal_quota_10(self.cfg.feature_dir,task,str(buffer),capacity=BUFFER_CAPACITY,
                seed=self.cfg.seed,drop_policy='random',rate=self.cfg.past_data_rate)
            write(planned,dict(task=task,files=sorted(p.name for p in buffer.glob('*.pt'))))
        checkpoint=Path(self.experiment_dir)/('mlrifa_%d_model_ep99.pth'%task)
        assert checkpoint.is_file()
        # Evidence: current trainable model is the stage's saved ep99 model.
        saved=torch.load(checkpoint,map_location='cpu')
        state=saved['state_dict'] if 'state_dict' in saved else saved
        assert state.keys()==self.policy.state_dict().keys()
        assert all(torch.equal(state[k].cpu(),v.detach().cpu()) for k,v in self.policy.state_dict().items()),'STAGE_EP99_MODEL_MISMATCH'
        expected={('t%d__'%task)+p.name for p in planned_cache(self.cfg.feature_dir,task,self.cfg.seed)}
        actual={p.name for p in buffer.glob('t%d__*.pt'%task)}
        assert expected==actual,'CACHE_MEMBERSHIP_CHANGED'
        self.cfg.buffer_dir=str(buffer)
        self.checkpoint(100,True)
        write(transaction/'COMMITTED.json',dict(task=task,buffer=str(buffer),plan_sha256=sha(planned)))

    def learn_all_tasks(self,datasets,benchmark,result_summary,current_task_id):
        self.a1_stage=current_task_id
        if self.resume and (current_task_id<self.resume['task'] or (current_task_id==self.resume['task'] and self.resume['stage_complete'])):return 0.,0.
        self.a1_start_epoch=self.resume['next_epoch'] if self.resume and self.resume['task']==current_task_id else 0
        if self.a1_start_epoch==100:
            ensure_stage(self,current_task_id)
            self.policy.load_state_dict(self.resume['state']['model'],strict=True)
            restore_rng(self.resume['state']['rng'])
            # Restore optimizer object through original stage bootstrap before saving.
            from .dataset import identified_dataset
            from torch.utils.data import DataLoader
            ds=identified_dataset(self.cfg.feature_dir,current_task_id,self.cfg.buffer_dir,self.cfg.seed,start=6)
            self.start_task(-1,DataLoader(ds,batch_size=self.cfg.train.batch_size))
            self.commit_stage();self.resume=None;return 0.,0.
        events=self.report/'events'
        if events.exists():
            for p in events.glob('task%d_epoch*.jsonl'%current_task_id):
                if '.interrupted' not in p.name and int(p.stem.split('epoch')[1])>=self.a1_start_epoch:
                    os.replace(p,p.with_name(p.name+'.interrupted.%d'%time.time_ns()))
        result=super().learn_all_tasks(datasets,benchmark,result_summary,current_task_id)
        self.resume=None
        return result
