"""Paired target-only adaptation, only after prospective signals are sealed."""
import argparse
import math
import os
import shutil
import time
from .common import OUT,STORE,read,write,sha
from . import runtime as rt


def main(seed,source,target):
    import torch
    from gradient_relation_experiment import kernel as g
    from cb_ifa_experiment import gpu_runtime as env
    sid='s%d_%d_%d'%(seed,source,target);dest=OUT/'transfers'/sid
    run=STORE/'transfers'/sid;dest.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    signal=OUT/'signals'/sid;seal=read(signal/'SEALED.json')
    assert sha(signal/'SIGNALS.json')==seal['signals_sha256'] and sha(signal/'SHADOW_ROLLBACK.json')==seal['shadow_sha256']
    shadow=read(signal/'SHADOW_ROLLBACK.json')
    stream_path=OUT/'sources'/('s%d_t%d'%(seed,target))/'STREAM.json'
    assert sha(stream_path)==shadow['target_stream_sha256']
    stream=read(stream_path);data=rt.load_data(seed,target,'A_train','train',source,target)
    algo=rt.build(seed,target,run)
    try:
        src=rt.model_path(seed,'source',source=source,step=1000)
        assert sha(src)==read(signal/'SIGNALS.json')['source_sha256']
        rt.load_model(algo,src);algo.policy.train()
        saved_initial=g.capture(algo)
        for key in ('model','optimizer','scheduler','optimizer_step','scheduler_step','training_modes'):
            assert g.digest(saved_initial[key])==shadow['initial_state_witness'][key],('Initial target state mismatch',key)
        recovery=STORE/'active_target_recovery.pt';position=0
        if recovery.exists():
            saved=torch.load(recovery,map_location='cpu')
            if saved['unit']==sid:
                assert saved['signal_seal_sha256']==sha(signal/'SEALED.json')
                g.restore(algo,saved['state']);assert all(g.compare_states(saved['state'],g.capture(algo)).values())
                position=saved['position']
            del saved
        del saved_initial
        begin=time.monotonic()
        for k in range(position,300):
            if shutil.disk_usage(STORE).free<2*1024**3:raise RuntimeError('STORAGE_RESERVE_REACHED')
            algo.policy.train();env.seed_all(stream['policy_seed'][k])
            lr=algo.optimizer.param_groups[0]['lr'];assert lr==stream['learning_rates'][k]
            loss=g.native(algo,rt.batch(data,stream['batches'][k]['indices']),target,stream['batches'][k]['epoch'])
            assert math.isfinite(loss)
            assert algo._optimizer_step_count==k+1 and algo._scheduler_step_count==k+1
            if k==0:
                after=g.capture(algo);checks={key:g.digest(v)==shadow['after_witness'][key] for key,v in after.items()}
                assert all(checks.values()),('Shadow/actual first step mismatch',checks)
                write(dest/'SHADOW_ACTUAL_FIRST_STEP_PARITY.json',dict(status='PASS',checks=checks,
                      source_signal_seal_sha256=sha(signal/'SEALED.json'),optimizer_updates=1,scheduler_updates=1,
                      formal_training_resumed=False));del after
            write(dest/'steps'/('%04d.json'%(k+1)),dict(step=k+1,record_indices=stream['batches'][k]['indices'],
                  policy_seed=stream['policy_seed'][k],lr=lr,lr_after=algo.optimizer.param_groups[0]['lr'],loss=float(loss),
                  optimizer_step=k+1,scheduler_step=k+1))
            if k+1 in (100,300):
                model=g.cpu_state(g.module(algo).state_dict())
                env.atomic_save(run/('step%04d.pt'%(k+1)),dict(model=model,seed=seed,source=source,target=target,step=k+1))
                env.atomic_save(recovery,dict(unit=sid,position=k+1,state=g.capture(algo),signal_seal_sha256=sha(signal/'SEALED.json')))
            if (k+1)%20==0:
                write(OUT/'STATUS.json',dict(stage='PILOT_TARGET_ADAPTATION',unit=sid,step=k+1,budget=300,
                      diagnostic_training_performed=True,formal_training_resumed=False))
        rows=[read(p) for p in sorted((dest/'steps').glob('*.json'))]
        rt.table(dest/'TRAINING_PROGRESS.csv',rows)
        direct=[read(OUT/'sources'/('s%d_t%d'%(seed,target))/'steps'/('%04d.json'%(k+1))) for k in range(300)]
        assert len(rows)==300
        assert all(r['record_indices']==d['record_indices'] and r['lr']==d['lr'] and r['lr_after']==d['lr_after'] for r,d in zip(rows,direct))
        write(dest/'STREAM_PARITY.json',dict(status='PASS',steps=300,target_stream_sha256=sha(stream_path),
              same_batch=True,same_lr=True,same_policy_rng=True,fresh_optimizer=True,
              reference300=str(rt.model_path(seed,'reference',source=target,step=300))))
        write(dest/'COMPLETE.json',dict(status='PASS',steps=300,seconds=time.monotonic()-begin,
              seed=seed,source=source,target=target,signal_seal_sha256=sha(signal/'SEALED.json'),
              diagnostic_training_performed=True,formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('seed','source','target'):p.add_argument('--'+k,type=int,required=True)
    a=p.parse_args();main(a.seed,a.source,a.target)
