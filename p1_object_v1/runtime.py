"""P1 loss insertion, immutable teachers, and epoch/stage transactional recovery."""
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
from .baseline_loop import P1EpochLoop
from .native_observe import NativeLossMixin
from .teacher_store import TeacherStore
from .policy_distill import forward_distribution,fields,sample_teacher,loss_from_samples
from .io import save,write,sha
from e1_object_v1.runtime import capture,restore
from e1_object_v1.control import rng_state,restore_rng
from libero.lifelong.algos.mlr import MLR,update_buffer_equal_quota_10,BUFFER_CAPACITY


class P1MLR(NativeLossMixin,P1EpochLoop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root=Path(self.cfg.p1_root);self.report=Path(self.cfg.p1_report)
        self.report.mkdir(parents=True,exist_ok=True)
        self.recovery=Path(self.cfg.p1_recovery)
        self.generator=torch.Generator(device='cpu').manual_seed(2000003+self.cfg.seed)
        self.resume=None;self.p1_stage=None;self.p1_start_epoch=0
        self.p1_totals=dict(steps=0,distill_steps=0,old_windows=0,negative_kl_windows=0,distill_forward_seconds=0.)
        manifest=None
        if self.recovery.exists():
            self.resume=torch.load(self.recovery,map_location='cpu')
            assert self.resume['protocol_sha256']==sha(self.cfg.p1_protocol),'P1_PROTOCOL_MISMATCH'
            assert self.resume['implementation_files']==self.implementation_files(),'P1_CODE_MISMATCH'
            self.cfg.buffer_dir=self.resume['buffer_dir'];manifest=self.resume['teacher_manifest']
            if manifest:assert sha(manifest)==self.resume['teacher_manifest_sha256']
            self.generator.set_state(self.resume['distill_rng'])
            self.p1_totals=self.resume['totals']
            self._retain_committed_logs()
        self.store=TeacherStore(self.root/'teacher_targets',self.cfg.seed,manifest)
        self.store.validate(self.cfg.buffer_dir)

    def implementation_files(self):
        return json.loads(Path(self.cfg.p1_protocol).read_text())['implementation_files']

    def _retain_committed_logs(self):
        for p in [Path(self.experiment_dir)/'lr_protocol.csv',Path(self.experiment_dir)/'paper_repro_train_curve.csv',self.report/'training_log.csv']:
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
            if self.resume['task']==self.p1_stage:restore(self,self.resume['state'])
            elif self.resume['task']<self.p1_stage and self.resume['stage_complete']:
                self.policy.load_state_dict(self.resume['state']['model'],strict=True)
                restore_rng(self.resume['state']['rng'])
            else:raise RuntimeError('Unexpected P1 recovery stage')
            self.generator.set_state(self.resume['distill_rng'])

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        # True native path when disabled: no new forward and no private sampling.
        if not self.cfg.p1_enabled or self.cfg.p1_beta==0:
            return MLR.observe(self,data,current_task_id,epoch,ifa_info)
        try:return NativeLossMixin.observe(self,data,current_task_id,epoch,ifa_info)
        except Exception as e:
            save(self.report/('FAILED_t%d_e%d_s%d.pt'%(current_task_id,epoch,self._optimizer_step_count)),
                 dict(error=repr(e),task=current_task_id,epoch=epoch,ids=list(data[2]),
                      distill_rng=self.generator.get_state(),optimizer_step=self._optimizer_step_count,
                      scheduler_step=self._scheduler_step_count))
            raise

    def finish_native(self,loss,data,task,epoch,base_bc):
        model=getattr(self.policy,'module',self.policy)
        tasks=[int(t) for t in data[1]];ids=list(data[2]);old=[i for i,t in enumerate(tasks) if t<task]
        assert all(6<=t<=task for t in tasks)
        native_loss=float(loss.detach());policy_value=0.;details={};elapsed=0.
        if old:
            torch.cuda.synchronize();begin=time.monotonic()
            teacher=self.store.lookup([ids[i] for i in old],[tasks[i] for i in old])
            actions,logp=sample_teacher(teacher,self.generator,16)
            old_data={k:v[old] for k,v in data[0].items()}
            student=forward_distribution(model,old_data)
            policy_loss,per=loss_from_samples(student,actions.to(loss.device),logp.to(loss.device))
            loss=loss+self.cfg.p1_beta*policy_loss
            sf=fields(student);pv=per.detach()
            details=dict(old_ids=[ids[i] for i in old],teacher_stages=[tasks[i] for i in old],
                teacher_hit_rate=1.,kl_min=float(pv.min()),kl_max=float(pv.max()),
                negative_kl_windows=int((pv<0).sum()),teacher_scale_mean=float(teacher['scales'].mean()),
                teacher_scale_min=float(teacher['scales'].min()),teacher_scale_max=float(teacher['scales'].max()),
                student_scale_mean=float(sf['scales'].detach().mean()),student_scale_min=float(sf['scales'].detach().min()),
                student_scale_max=float(sf['scales'].detach().max()),teacher_weight_max=float(teacher['log_weights'].exp().max()),
                student_weight_max=float(sf['log_weights'].detach().exp().max()))
            policy_value=float(policy_loss.detach())
            torch.cuda.synchronize();elapsed=time.monotonic()-begin
        assert bool(torch.isfinite(loss)),'NONFINITE_TOTAL_LOSS'
        before=(self._optimizer_step_count,self._scheduler_step_count)
        lr=float(self.optimizer.param_groups[0]['lr'])
        (self.loss_scale*loss).backward()
        if self.cfg.train.grad_clip is not None:
            gn=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
        else:gn=torch.sqrt(sum(p.grad.square().sum() for p in self.policy.parameters() if p.grad is not None))
        assert bool(torch.isfinite(gn)),'NONFINITE_GRADIENT'
        self.optimizer_step_and_maybe_schedule()
        assert self._optimizer_step_count==before[0]+1
        assert self._scheduler_step_count==before[1]+int(self.scheduler is not None and self.repro_scheduler_mode=='batch_step')
        row=dict(seed=self.cfg.seed,task=task,epoch=epoch,step=self._optimizer_step_count,
            bc=float(base_bc),ifa_weighted=native_loss-float(base_bc),native_loss=native_loss,
            policy_kl=policy_value,policy_weighted=self.cfg.p1_beta*policy_value,total_loss=float(loss.detach()),
            old_count=len(old),lr=lr,gradient_norm=float(gn),gradient_clipped=bool(self.cfg.train.grad_clip is not None and gn>self.cfg.train.grad_clip),
            distill_forward_seconds=elapsed,**details)
        path=self.report/'events'/('task%d_epoch%03d.jsonl'%(task,epoch));path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.p1_totals['steps']+=1;self.p1_totals['distill_steps']+=bool(old)
        self.p1_totals['old_windows']+=len(old)
        self.p1_totals['negative_kl_windows']+=details.get('negative_kl_windows',0)
        self.p1_totals['distill_forward_seconds']+=elapsed
        return loss.item()

    def checkpoint(self,next_epoch,stage_complete=False):
        manifest=self.store.manifest
        value=dict(state=capture(self),task=self.p1_stage,next_epoch=next_epoch,stage_complete=stage_complete,
            lifecycle='BUFFER_AND_TARGETS_COMMITTED' if stage_complete else ('EP99_SAVED' if next_epoch==100 else 'TRAINING'),
            distill_rng=self.generator.get_state(),totals=self.p1_totals.copy(),buffer_dir=self.cfg.buffer_dir,
            buffer_files=sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt')),
            teacher_manifest=str(manifest) if manifest else None,teacher_manifest_sha256=sha(manifest) if manifest else None,
            protocol_sha256=sha(self.cfg.p1_protocol),implementation_files=self.implementation_files())
        save(self.recovery,value)
        write(self.report/'STATUS.json',dict(status=value['lifecycle'],task=self.p1_stage,next_epoch=next_epoch,
              seed=self.cfg.seed,optimizer_step=self._optimizer_step_count,scheduler_step=self._scheduler_step_count,
              totals=self.p1_totals,recovery=str(self.recovery),p1_training_active=not stage_complete,e1_enabled=False))

    def _append_csv(self,path,fieldnames,row):
        MLR._append_csv(path,fieldnames,row)
        if Path(path).name=='paper_repro_train_curve.csv':
            events=self.report/'events'/('task%d_epoch%03d.jsonl'%(row['stage'],row['epoch']))
            rows=[json.loads(s) for s in events.read_text().splitlines()]
            summary=dict(stage=row['stage'],epoch=row['epoch'],steps=len(rows),
                **{k:sum(r[k] for r in rows)/len(rows) for k in ['bc','ifa_weighted','policy_kl','policy_weighted','total_loss','gradient_norm']},
                old_windows=sum(r['old_count'] for r in rows),negative_kl_windows=sum(r.get('negative_kl_windows',0) for r in rows),
                distill_forward_seconds=sum(r['distill_forward_seconds'] for r in rows),lr_end=row['lr_end'])
            MLR._append_csv(str(self.report/'training_log.csv'),list(summary),summary)
            self.checkpoint(int(row['epoch'])+1)

    def commit_stage(self):
        task=self.p1_stage;transaction=self.root/'transactions'/('task%d'%task)
        transaction.mkdir(parents=True,exist_ok=True)
        planned=transaction/'PLAN.json';buffer=transaction/'buffer'
        # Work only in P1-owned staging storage. Completed selection is persisted
        # before any teacher work; recovery never resamples a completed plan.
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
        assert all(torch.equal(state[k].cpu(),v.detach().cpu()) for k,v in self.policy.state_dict().items()),'TEACHER_EP99_MODEL_MISMATCH'
        manifest=self.store.build_stage(getattr(self.policy,'module',self.policy),buffer,task,checkpoint,self.report)
        self.cfg.buffer_dir=str(buffer)
        self.checkpoint(100,True)
        write(self.report.parent/('TEACHER_MANIFEST_seed%d.json'%self.cfg.seed),json.loads(manifest.read_text()))
        write(transaction/'COMMITTED.json',dict(task=task,manifest=str(manifest),sha256=sha(manifest)))

    def learn_all_tasks(self,datasets,benchmark,result_summary,current_task_id):
        self.p1_stage=current_task_id
        if self.resume and (current_task_id<self.resume['task'] or (current_task_id==self.resume['task'] and self.resume['stage_complete'])):return 0.,0.
        self.p1_start_epoch=self.resume['next_epoch'] if self.resume and self.resume['task']==current_task_id else 0
        self.store.validate(self.cfg.buffer_dir)
        if self.p1_start_epoch==100:
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
                if '.interrupted' not in p.name and int(p.stem.split('epoch')[1])>=self.p1_start_epoch:
                    os.replace(p,p.with_name(p.name+'.interrupted.%d'%time.time_ns()))
        result=super().learn_all_tasks(datasets,benchmark,result_summary,current_task_id)
        self.resume=None
        return result
