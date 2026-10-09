"""G1 loss insertion, immutable buffers, and epoch/stage transactional recovery."""
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
from .baseline_loop import G1EpochLoop
from .native_observe import NativeLossMixin
from .io import save,write,sha
from .state import capture,restore,rng_state,restore_rng
from .gate import select_loss
from libero.lifelong.algos.mlr import MLR,update_buffer_equal_quota_10,BUFFER_CAPACITY


class G1MLR(NativeLossMixin,G1EpochLoop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root=Path(self.cfg.g1_root);self.report=Path(self.cfg.g1_report)
        self.report.mkdir(parents=True,exist_ok=True)
        self.recovery=Path(self.cfg.g1_recovery)
        self.resume=None;self.g1_stage=None;self.g1_start_epoch=0
        self.g1_totals=dict(steps=0,ifa_constructed=0,nonzero_ifa=0,decisions=0,closed=0,zero_gradient=0,ifa_proposed=0.,ifa_retained=0.,probe_seconds=0.,dot_seconds=0.)
        if self.recovery.exists():
            self.resume=torch.load(self.recovery,map_location='cpu')
            assert self.resume['protocol_sha256']==sha(self.cfg.g1_protocol),'G1_PROTOCOL_MISMATCH'
            assert self.resume['implementation_files']==self.implementation_files(),'G1_CODE_MISMATCH'
            self.cfg.buffer_dir=self.resume['buffer_dir']
            assert sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt'))==self.resume['buffer_files']
            self.g1_totals=self.resume['totals']
            self._retain_committed_logs()

    def implementation_files(self):
        return json.loads(Path(self.cfg.g1_protocol).read_text())['implementation_files']

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
            if self.resume['task']==self.g1_stage:restore(self,self.resume['state'])
            elif self.resume['task']<self.g1_stage and self.resume['stage_complete']:
                self.policy.load_state_dict(self.resume['state']['model'],strict=True)
                restore_rng(self.resume['state']['rng'])
            else:raise RuntimeError('Unexpected G1 recovery stage')

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        # True native path when disabled: no new forward and no private sampling.
        if not self.cfg.g1_enabled:
            return MLR.observe(self,data,current_task_id,epoch,ifa_info)
        try:return NativeLossMixin.observe(self,data,current_task_id,epoch,ifa_info)
        except Exception as e:
            save(self.report/('FAILED_t%d_e%d_s%d.pt'%(current_task_id,epoch,self._optimizer_step_count)),
                 dict(error=repr(e),task=current_task_id,epoch=epoch,ids=list(data[2]),
                      failed_state_not_recovery=capture(self),data=data,
                      optimizer_step=self._optimizer_step_count,scheduler_step=self._scheduler_step_count))
            raise

    def finish_native(self,loss,data,task,epoch,bc_loss,aux_terms):
        assert torch.distributed.get_world_size()==1
        model=getattr(self.policy,'module',self.policy)
        params=tuple(p for p in model.temporal_transformer.parameters() if p.requires_grad)
        selected,gate=select_loss(bc_loss,loss,aux_terms,params,getattr(self,'smoke_force',None))
        before=(self._optimizer_step_count,self._scheduler_step_count)
        lr=float(self.optimizer.param_groups[0]['lr'])
        (self.loss_scale*selected).backward()
        if self.cfg.train.grad_clip is not None:
            gn=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
        else:gn=torch.sqrt(sum(p.grad.square().sum() for p in self.policy.parameters() if p.grad is not None))
        assert bool(torch.isfinite(gn)), 'NONFINITE_FINAL_GRADIENT'
        self.optimizer_step_and_maybe_schedule()
        assert self._optimizer_step_count==before[0]+1
        assert self._scheduler_step_count==before[1]+int(self.scheduler is not None and self.repro_scheduler_mode=='batch_step')
        tasks=[int(t) for t in data[1]]
        row=dict(seed=self.cfg.seed,task=task,epoch=epoch,step=self._optimizer_step_count,
            current_count=tasks.count(task),old_count=sum(t<task for t in tasks),
            ifa_sample_count=tasks.count(task) if aux_terms else 0,bc=float(bc_loss.detach()),
            native_loss=float(loss.detach()),total_loss=float(selected.detach()),lr=lr,
            gradient_norm=float(gn),peak_gpu_bytes=torch.cuda.max_memory_allocated(),**gate)
        self.last_row=row
        path=self.report/'events'/('task%d_epoch%03d.jsonl'%(task,epoch));path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        totals=self.g1_totals
        for k,value in dict(steps=1,ifa_constructed=int(gate['has_ifa']),nonzero_ifa=int(gate['has_ifa'] and not gate['hinge_zero']),
            decisions=int(gate['dot'] is not None),closed=int(gate['w']==0),zero_gradient=int(gate['reason']=='ZERO_GRADIENT'),
            ifa_proposed=sum(gate['weighted_terms']),ifa_retained=gate['retained_ifa'],probe_seconds=gate['probe_seconds'],dot_seconds=gate['dot_seconds']).items():totals[k]+=value
        return selected.item()

    def checkpoint(self,next_epoch,stage_complete=False):
        value=dict(state=capture(self),task=self.g1_stage,next_epoch=next_epoch,stage_complete=stage_complete,
            lifecycle='BUFFER_COMMITTED' if stage_complete else ('EP99_SAVED' if next_epoch==100 else 'TRAINING'),
            totals=self.g1_totals.copy(),buffer_dir=self.cfg.buffer_dir,
            buffer_files=sorted(p.name for p in Path(self.cfg.buffer_dir).glob('*.pt')),
            protocol_sha256=sha(self.cfg.g1_protocol),implementation_files=self.implementation_files())
        save(self.recovery,value)
        write(self.report/'STATUS.json',dict(status=value['lifecycle'],task=self.g1_stage,next_epoch=next_epoch,
              seed=self.cfg.seed,optimizer_step=self._optimizer_step_count,scheduler_step=self._scheduler_step_count,
              totals=self.g1_totals,recovery=str(self.recovery),g1_training_active=not stage_complete,e1_enabled=False))

    def _append_csv(self,path,fieldnames,row):
        MLR._append_csv(path,fieldnames,row)
        if Path(path).name=='paper_repro_train_curve.csv':
            events=self.report/'events'/('task%d_epoch%03d.jsonl'%(row['stage'],row['epoch']))
            rows=[json.loads(s) for s in events.read_text().splitlines()]
            summary=dict(stage=row['stage'],epoch=row['epoch'],steps=len(rows),
                **{k:sum(r[k] for r in rows)/len(rows) for k in ['bc','native_loss','retained_ifa','total_loss','gradient_norm']},
                old_windows=sum(r['old_count'] for r in rows),closed=sum(r['w']==0 for r in rows),
                probe_seconds=sum(r['probe_seconds'] for r in rows),lr_end=row['lr_end'])
            MLR._append_csv(str(self.report/'training_log.csv'),list(summary),summary)
            self.checkpoint(int(row['epoch'])+1)

    def commit_stage(self):
        task=self.g1_stage;transaction=self.root/'transactions'/('task%d'%task)
        transaction.mkdir(parents=True,exist_ok=True)
        planned=transaction/'PLAN.json';buffer=transaction/'buffer'
        # Work only in G1-owned staging storage. Completed selection is persisted
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
        self.cfg.buffer_dir=str(buffer)
        self.checkpoint(100,True)
        write(transaction/'COMMITTED.json',dict(task=task,buffer=str(buffer),plan_sha256=sha(planned)))

    def learn_all_tasks(self,datasets,benchmark,result_summary,current_task_id):
        self.g1_stage=current_task_id
        if self.resume and (current_task_id<self.resume['task'] or (current_task_id==self.resume['task'] and self.resume['stage_complete'])):return 0.,0.
        self.g1_start_epoch=self.resume['next_epoch'] if self.resume and self.resume['task']==current_task_id else 0
        if self.g1_start_epoch==100:
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
                if '.interrupted' not in p.name and int(p.stem.split('epoch')[1])>=self.g1_start_epoch:
                    os.replace(p,p.with_name(p.name+'.interrupted.%d'%time.time_ns()))
        result=super().learn_all_tasks(datasets,benchmark,result_summary,current_task_id)
        self.resume=None
        return result
