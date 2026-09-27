"""Native source training; its identical first300 calls supply base references."""
import argparse
import math
import os
import shutil
import time
import numpy as np
from .common import *
from . import confirm_runtime as rt
from .measure import extract

def main(seed,task):
    import torch
    from cb_ifa_experiment import gpu_runtime as env
    from gradient_relation_experiment import kernel as g
    from torch.utils.data.distributed import DistributedSampler
    report=rt.REPORT/'sources'/('s%d_t%d'%(seed,task));run=rt.DATA/'sources'/report.name
    report.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    if (report/'COMPLETE.json').exists():return
    for p,h in read(rt.REPORT/'FROZEN_INPUTS.json')['files'].items():assert sha(ROOT/p)==h
    data=rt.load(seed,task,'A_train','train');dd=rt.load(seed,task,'D_signal','signal')
    n=len(data['rows']);proto=read(rt.REPORT/'TRAINING_PROTOCOL.json');bs=proto['batch_size'];bpe=math.ceil(n/bs)
    sampler=DistributedSampler(range(n),num_replicas=1,rank=0,shuffle=True,seed=seed,drop_last=False)
    batches=[]
    for epoch in range(math.ceil(1000/bpe)):
        sampler.set_epoch(epoch);indices=list(sampler)
        batches.extend(dict(epoch=epoch,indices=indices[i:i+bs]) for i in range(0,n,bs))
    stream=dict(seed=seed,target=task,batches=batches[:1000],policy_seed=[stable_seed('task_relation_signal_screen_v2','train',seed,task,k) for k in range(1000)],learning_rates=proto['step_lr'][:1000])
    freeze(report/'STREAM.json',stream)
    algo=rt.build(seed,task,run)
    try:
        initial=g.capture(algo)
        freeze(report/'INITIALIZATION.json',dict(model_sha256=g.digest(initial['model']),optimizer_initial_sha256=g.digest(initial['optimizer']),scheduler_initial_sha256=g.digest(initial['scheduler']),state_sha256={k:g.digest(v) for k,v in initial.items()},trainable_parameters=[n for n,p in algo.policy.named_parameters() if p.requires_grad],configuration_template_sha256=sha(ROOT/'runs/fixed_kappa_confirmation_v2/formal/seed45/mlr/run_001/config.json'),formal_training_resumed=False))
        def endpoint(call):
            model=g.cpu_state(g.module(algo).state_dict());path=run/('step%04d.pt'%call)
            if call==0:
                shared=rt.model_path(seed,'base');shared.parent.mkdir(parents=True,exist_ok=True)
                if shared.exists():assert g.digest(torch.load(shared,map_location='cpu')['model'])==g.digest(model)
                else:env.atomic_save(shared,dict(model=model,seed=seed,call=0))
                if not path.exists():os.link(shared,path)
            else:env.atomic_save(path,dict(model=model,seed=seed,source=task,call=call))
            write(report/('MODEL_%04d.json'%call),dict(path=str(path),sha256=sha(path),model_sha256=g.digest(model),call=call))
        recovery=rt.DATA/'active_source_recovery.pt';position=0;effective=0
        if recovery.exists():
            saved=torch.load(recovery,map_location='cpu')
            if saved['seed']==seed and saved['task']==task:
                assert saved['stream_sha256']==sha(report/'STREAM.json')
                g.restore(algo,saved['state']);assert all(g.compare_states(saved['state'],g.capture(algo)).values())
                position=saved['position'];effective=saved['effective']
            del saved
        if position==0:
            # Same native code and first call twice; discard both, then start real trajectory.
            witnesses=[]
            for _ in range(2):
                g.restore(algo,initial);algo.policy.train();env.seed_all(stream['policy_seed'][0])
                g.native(algo,rt.batch(data,stream['batches'][0]['indices']),task,0);witnesses.append(g.capture(algo))
            checks=g.compare_states(*witnesses);assert all(checks.values())
            g.restore(algo,initial);assert all(g.compare_states(initial,g.capture(algo)).values())
            write(report/'NATIVE_PARITY.json',dict(status='PASS',checks=checks,first_lr=proto['step_lr'][0],discarded_zero_lr_calls=2))
            endpoint(0)
            write(report/'D_PREFIX_0000.json',dict(call=0,effective=0,nll=extract(algo,dd)['nll'].tolist()))
            del witnesses
        del initial
        begin=time.monotonic()
        for k in range(position,1000):
            if shutil.disk_usage(rt.DATA).free<2*1024**3:raise RuntimeError('STORAGE_RESERVE')
            algo.policy.train();env.seed_all(stream['policy_seed'][k]);lr=algo.optimizer.param_groups[0]['lr']
            assert lr==stream['learning_rates'][k]
            before=tuple(p.detach().clone() for p in algo.policy.parameters() if p.requires_grad) if k<50 else None
            loss=float(g.native(algo,rt.batch(data,stream['batches'][k]['indices']),task,stream['batches'][k]['epoch']))
            assert math.isfinite(loss) and algo._optimizer_step_count==algo._scheduler_step_count==k+1
            dn=None
            if before is not None:
                dn=g.norm(tuple(p.detach()-v for p,v in zip((p for p in algo.policy.parameters() if p.requires_grad),before)))
                if lr>0 and dn>0:effective+=1
                del before
                state=g.capture(algo);nll=extract(algo,dd)['nll'];assert all(g.compare_states(state,g.capture(algo)).values())
                write(report/('D_PREFIX_%04d.json'%(k+1)),dict(call=k+1,effective=effective,nll=nll.tolist(),state_sha256={f:g.digest(v) for f,v in state.items()}));del state
            write(report/'steps'/('%04d.json'%(k+1)),dict(call=k+1,record_indices=stream['batches'][k]['indices'],policy_seed=stream['policy_seed'][k],lr=lr,lr_after=algo.optimizer.param_groups[0]['lr'],loss=loss,d_norm=dn,effective_prefix=effective,optimizer_step=k+1,scheduler_step=k+1))
            if k+1 in (100,300,1000):endpoint(k+1)
            if (k+1)%100==0:env.atomic_save(recovery,dict(seed=seed,task=task,position=k+1,effective=effective,state=g.capture(algo),stream_sha256=sha(report/'STREAM.json')))
            if (k+1)%20==0:
                write(rt.REPORT/'STATUS.json',dict(stage='SOURCE_TRAINING',seed=seed,task=task,call=k+1,budget=1000,seconds=time.monotonic()-begin,formal_training_resumed=False,diagnostic_training_performed=True))
                print('SOURCE',seed,task,k+1,loss,round(time.monotonic()-begin,2),flush=True)
        write(report/'REFERENCE_PREFIX_PARITY.json',dict(status='PASS_BY_COMMON_TRAJECTORY',reference_call=300,common_horizon=proto['common_schedule_horizon'],stream_sha256=sha(report/'STREAM.json'),fresh_optimizer=True,reference_model_sha256=sha(run/'step0300.pt'),explanation='The reference is the actual first300 calls of this base-initialized source run; target adapters verify the same initial optimizer/scheduler and prefix stream.',formal_training_resumed=False))
        write(report/'COMPLETE.json',dict(status='PASS',seed=seed,task=task,calls=1000,seconds=time.monotonic()-begin,model_sha256=sha(run/'step1000.pt'),formal_training_resumed=False,diagnostic_training_performed=True))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
    access=rt.data_firewall()
    try:main(a.seed,a.task)
    finally:write(rt.REPORT/'access'/('source_s%d_t%d.json'%(a.seed,a.task)),dict(paths=sorted(access),formal_training_resumed=False))
