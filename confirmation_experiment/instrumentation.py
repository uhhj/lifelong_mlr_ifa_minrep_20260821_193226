"""Observe the frozen native steps, writing scalars and pre-update witnesses.

The historical observe/projector/clip/optimizer/scheduler functions are called
unchanged. An instance-local context temporarily wraps the projector, never its
decision or arithmetic. Diagnostic n signals are algebraic, not extra backward.
"""
import hashlib
import json
import math
import os
import struct
import time
from collections import Counter
from pathlib import Path

import torch
from libero.lifelong.algos.mlr import MLR,parse_path
from libero.lifelong.algos.mlr_fixed_kappa import MLRFixedKappa
import libero.lifelong.algos.fixed_kappa as core
from confirmation_experiment.common import (ROOT,RUNS,REPORTS,write,sha,read,rng_state,state_to_cpu,
                                           state_digest,atomic_torch_save,wait_for)
from confirmation_experiment.protocol import KAPPA,derived_signals,snapshot_slot,epoch_phase

def grad_norm(parameters):
    squares=torch.zeros((),device=next(iter(parameters)).device,dtype=torch.float64)
    for p in parameters:
        if p.grad is not None:squares+=(p.grad.detach()*p.grad.detach()).sum(dtype=torch.float64)
    return math.sqrt(float(squares))

def origin_for_record(path,feature_root):
    path=Path(path);task_id=int(parse_path(str(path)))
    name=path.stem.split('__')[1]+path.suffix if path.name.startswith('t') else path.name
    return Path(feature_root)/str(task_id)/name

class TraceMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.confirmation_root=Path(self.experiment_dir).parent
        self.event_logging=os.environ.get('CONFIRMATION_EVENT_LOGGING','1')=='1'
        self.snapshot_logging=os.environ.get('CONFIRMATION_SNAPSHOTS','1')=='1'
        self.integrity_barriers=os.environ.get('CONFIRMATION_BARRIERS','1')=='1'
        self._confirm_global=0;self._confirm_epoch_key=None;self._confirm_batch=0
        self._confirm_trace=hashlib.sha256();self._confirm_trace_count=0
        self._confirm_slots=set();self._confirm_snapshot_rows=[];self._confirm_hook_metrics=None
        self._confirm_events=None;self._confirm_stage=6;self._confirm_plan=None
        self._confirm_epoch_indices=None;self._confirm_buffer_before=None
        self._confirm_timings={'native_observe_seconds':0.,'instrumentation_seconds':0.,'snapshot_seconds':0.,'steps':0}

    def start_task(self,task,train_loader):
        result=super().start_task(task,train_loader)
        self._confirm_loader=train_loader
        ds=getattr(train_loader,'dataset',None)
        if ds is not None and hasattr(ds,'file_paths'):
            sampler=train_loader.sampler
            hashes=[sha(p) for p in ds.file_paths]
            origins=[]
            for p,h in zip(ds.file_paths,hashes):
                p=Path(p);origin=origin_for_record(p,self.cfg.feature_dir)
                assert origin.is_file() and sha(origin)==h,('Replay origin cannot be restored',str(p))
                origins.append(str(origin.relative_to(ROOT)))
            plan={'task':self._confirm_stage,'records':[str(Path(p).relative_to(ROOT)) for p in ds.file_paths],
                  'record_sha256':hashes,'immutable_feature_origins':origins,
                  'sampler':{'class':type(sampler).__name__,'seed':int(sampler.seed),
                             'rank':int(sampler.rank),'num_replicas':int(sampler.num_replicas),
                             'shuffle':bool(sampler.shuffle),'drop_last':bool(sampler.drop_last)},
                  'batch_size':int(train_loader.batch_size),'drop_last':bool(train_loader.drop_last),
                  'num_workers':int(train_loader.num_workers),'batches_per_epoch':len(train_loader),
                  'epochs':int(self.cfg.train.n_epochs),
                  'recovery':'Store current epoch permutation in each snapshot; future epochs use the stored DistributedSampler definition. Resume remaining current-epoch records without creating a fresh DataLoader iterator/RNG draw; next epoch uses the original iterator creation.'}
            assert plan['num_workers']==0 and plan['sampler']['num_replicas']==1
            path=self.confirmation_root/f'plans/task{self._confirm_stage}.json'
            write(path,plan);self._confirm_plan=plan;self._confirm_plan_path=path
        return result

    def learn_all_tasks(self,datasets,benchmark,result_summary,current_task_id):
        self._confirm_stage=int(current_task_id)
        result=super().learn_all_tasks(datasets,benchmark,result_summary,current_task_id)
        if self._confirm_events:self._confirm_events.flush()
        write(self.confirmation_root/'snapshot_index.json',{'max_mechanism_snapshots_per_arm':16,
              'saved':self._confirm_snapshot_rows,'slots_saved':sorted(self._confirm_slots)})
        write(self.confirmation_root/'instrumentation_timing.json',self._confirm_timings)
        if current_task_id==6:
            witness={'task':6,'batches':self._confirm_trace_count,'loss_sequence_sha256':self._confirm_trace.hexdigest(),
                     'model_state_sha256':state_digest(self.policy.state_dict()),
                     'optimizer_sha256':state_digest(self.optimizer.state_dict()),
                     'scheduler_sha256':state_digest(self.scheduler.state_dict() if self.scheduler else None),
                     'rng_sha256':state_digest(rng_state()),'optimizer_step':self._optimizer_step_count}
            write(self.confirmation_root/'TASK6_TRAIN_READY.json',witness)
            if self.integrity_barriers:
                wait_for(self.confirmation_root/'RELEASE_TRAIN',RUNS/'FAILED.json')
        write(self.confirmation_root/'TRAIN_PROGRESS.json',{'seed':int(self.cfg.seed),'arm':self.confirmation_arm,
              'task':int(current_task_id),'epoch':99,'stage_complete':True,'global_optimizer_step':self._confirm_global})
        return result

    def _save_snapshot(self,metrics):
        if not self.snapshot_logging or not self._confirm_plan:return
        remaining=self._confirm_plan['epochs']*self._confirm_plan['batches_per_epoch']-self._optimizer_step_count
        slot=snapshot_slot(self._confirm_stage,self._confirm_epoch,metrics['projected'],metrics['z'],self._confirm_slots,remaining)
        if slot is None:return
        begin=time.perf_counter()
        model=state_to_cpu(self.policy.state_dict())
        # Parameters have not moved. Replace any forward-mutated buffers with
        # their captured pre-forward values so the witness restores that boundary.
        for k,v in self._confirm_buffer_before.items():model[k]=v
        run_cfg=self.confirmation_root/'run_001/config.json'
        cfg_hash=sha(run_cfg) if run_cfg.exists() else state_digest(dict(self.cfg))
        frozen=REPORTS/'FROZEN_INPUTS.json'
        state={'format':'confirmation_pre_forward_v1','model':model,
               'optimizer':state_to_cpu(self.optimizer.state_dict()),
               'scheduler':state_to_cpu(self.scheduler.state_dict()) if self.scheduler else None,
               'optimizer_step':self._optimizer_step_count,'scheduler_step':self._scheduler_step_count,
               'global_optimizer_step':self._confirm_global,'rng':self._confirm_rng_before,
               'training_modes':{n:m.training for n,m in self.policy.named_modules()},
               'seed':int(self.cfg.seed),'current_task':self._confirm_stage,'epoch':self._confirm_epoch,
               'batch_index_in_epoch':self._confirm_batch,'batch':state_to_cpu(self._confirm_input),
               'record_indices':self._confirm_record_indices,'epoch_sampler_indices':self._confirm_epoch_indices,
               'stage_plan_path':str(self._confirm_plan_path.relative_to(ROOT)),
               'stage_plan_sha256':sha(self._confirm_plan_path),'config':dict(self.cfg),'config_sha256':cfg_hash,
               'frozen_inputs_sha256':sha(frozen) if frozen.exists() else None,
               'implementation_sha':os.environ.get('CONFIRMATION_IMPLEMENTATION_SHA','PRECHECK'),
               'slot':slot,'selection_metrics':metrics,
               'restore_boundary':'Immediately before native observe/map-to-device/zero_grad/forward; saved gradients are unnecessary because native observe clears them.'}
        dest=self.confirmation_root/'snapshots'/f'{slot}.pt'
        atomic_torch_save(dest,state)
        self._confirm_slots.add(slot)
        self._confirm_snapshot_rows.append({'slot':slot,'path':str(dest.relative_to(ROOT)),
              'sha256':sha(dest),'epoch':self._confirm_epoch,'batch':self._confirm_batch,
              'global_optimizer_step':self._confirm_global,'trigger':metrics['projected'],'z':metrics['z']})
        write(self.confirmation_root/'snapshot_index.json',{'max_mechanism_snapshots_per_arm':16,'saved':self._confirm_snapshot_rows})
        self._confirm_timings['snapshot_seconds']+=time.perf_counter()-begin

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        begin=time.perf_counter()
        self._confirm_stage=int(current_task_id);self._confirm_epoch=int(epoch)
        key=(int(current_task_id),int(epoch))
        if key!=self._confirm_epoch_key:
            self._confirm_epoch_key=key;self._confirm_batch=0
            if self._confirm_plan:
                self._confirm_epoch_indices=list(iter(self._confirm_loader.sampler))
        ids=[int(x) for x in data[1]];n_old=sum(x<int(current_task_id) for x in ids);n_new=len(ids)-n_old
        if self._confirm_plan:
            lo=self._confirm_batch*self._confirm_plan['batch_size']
            self._confirm_record_indices=self._confirm_epoch_indices[lo:lo+len(ids)]
            paths=[ROOT/self._confirm_plan['records'][i] for i in self._confirm_record_indices]
            assert [int(parse_path(str(p))) for p in paths]==ids
        else:self._confirm_record_indices=[]
        candidate=self.snapshot_logging and current_task_id in (7,8) and epoch>=30
        self._confirm_input=data
        self._confirm_rng_before=rng_state() if candidate else None
        self._confirm_buffer_before={n:state_to_cpu(v) for n,v in self.policy.named_buffers()} if candidate else {}
        stage_step=self._optimizer_step_count;lr=float(self.optimizer.param_groups[0]['lr'])
        original=core.project_parameter_grads_
        self._confirm_hook_metrics=None
        def observed_project(parameters,reference,*args,**kwargs):
            params=tuple(parameters)
            # Only transient copies. No gradient tensor is written to the log.
            before=[p.grad.detach().clone() if p.grad is not None else None for p in params]
            m=original(params,reference,*args,**kwargs)
            m.update(derived_signals(m,n_new,n_old))
            actual2=torch.zeros((),device=params[0].device,dtype=torch.float64)
            if m['projected']:
                for p,old in zip(params,before):
                    if p.grad is None:continue
                    delta=p.grad.detach() if old is None else p.grad.detach()-old
                    actual2+=(delta*delta).sum(dtype=torch.float64)
            m['requested_correction_norm']=(KAPPA*abs(m['dot_before'])/m['reference_grad_norm']) if m['projected'] else 0.
            m['actual_written_correction_norm']=math.sqrt(float(actual2))
            m['actual_correction_ratio']=m['actual_written_correction_norm']/m['mixed_grad_norm'] if m['mixed_grad_norm'] else 0.
            self._confirm_hook_metrics=m
            if candidate:self._save_snapshot(m)
            return m
        if self.confirmation_arm=='soft' and self.event_logging:core.project_parameter_grads_=observed_project
        native_begin=time.perf_counter()
        try:loss=super().observe(data,current_task_id,epoch,ifa_info)
        finally:core.project_parameter_grads_=original
        native_seconds=time.perf_counter()-native_begin
        self._confirm_global+=1
        if current_task_id==6:
            self._confirm_trace.update(struct.pack('!d',float(loss)));self._confirm_trace_count+=1
        if self.confirmation_arm=='soft' and self.event_logging:
            m=self._confirm_hook_metrics
            if m is None:raise RuntimeError('Native projector observer was not called')
            native=self._rc_rows[-1]
            event=dict(m,seed=int(self.cfg.seed),current_task=int(current_task_id),epoch=int(epoch),
                       trigger=bool(m['projected']),
                       phase=epoch_phase(epoch),batch_index_in_epoch=self._confirm_batch,
                       global_optimizer_step_before=self._confirm_global-1,global_optimizer_step=self._confirm_global,
                       optimizer_step_before=stage_step,optimizer_step=self._optimizer_step_count,
                       lr=lr,lr_after=float(self.optimizer.param_groups[0]['lr']),n_new=n_new,n_old=n_old,
                       task_source_counts=dict(Counter(map(str,ids))),record_indices=self._confirm_record_indices,
                       mixed_bc_loss=float(loss),new_bc_loss=native['new_loss'],old_bc_loss=native['old_loss'],
                       preclip_gradient_norm=native['preclip_grad_norm'],postclip_gradient_norm=grad_norm(tuple(self.policy.parameters())),
                       clip_max_norm=float(self.cfg.train.grad_clip),
                       native_observe_with_logging_seconds=native_seconds)
            if self._confirm_events is None:
                self._confirm_events=(self.confirmation_root/'events.jsonl').open('x',encoding='utf-8',buffering=1024*1024)
            self._confirm_events.write(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n')
            if self._confirm_batch%100==0:self._confirm_events.flush()
        self._confirm_batch+=1
        if self._confirm_batch%100==0:
            write(self.confirmation_root/'TRAIN_PROGRESS.json',{'seed':int(self.cfg.seed),'arm':self.confirmation_arm,
                  'task':int(current_task_id),'epoch':int(epoch),'batch':self._confirm_batch,
                  'global_optimizer_step':self._confirm_global})
        self._confirm_timings['steps']+=1
        self._confirm_timings['native_observe_seconds']+=native_seconds
        self._confirm_timings['instrumentation_seconds']+=time.perf_counter()-begin-native_seconds
        return loss

class ConfirmationMLR(TraceMixin,MLR):
    confirmation_arm='mlr'

class ConfirmationSoft(TraceMixin,MLRFixedKappa):
    confirmation_arm='soft'
