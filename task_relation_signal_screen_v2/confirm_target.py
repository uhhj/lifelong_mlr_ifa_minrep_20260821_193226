"""One native300-call trajectory, with Cq sealed at its real prefix boundary."""
import argparse
import math
import time
import shutil
import numpy as np
from .common import *
from . import confirm_runtime as rt
from .measure import extract,ids,mean_optional
from .numeric import avg,tail
from .confirm_signals import predict

def main(seed,source,target):
    import torch
    from cb_ifa_experiment import gpu_runtime as env
    from gradient_relation_experiment import kernel as g
    sid=unit(seed,source,target);report=rt.REPORT/'transfers'/sid;run=rt.DATA/'transfers'/sid
    report.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    if (report/'COMPLETE.json').exists():return
    signal=rt.REPORT/'signals'/sid;seal=read(signal/'C0_SEALED.json');assert sha(signal/'C0.json')==seal['signals_sha256']
    c0=read(signal/'C0.json');assert sha(c0['window_path'])==c0['window_sha256']
    assert c0['source_checkpoint_sha256']==sha(rt.model_path(seed,'source',source))
    with np.load(c0['window_path']) as z:start_old=z['source_source'].copy();start_new=z['source_target'].copy();base_new=z['base_target'].copy()
    data=rt.load(seed,target,'A_train','train');sd=rt.load(seed,source,'D_signal','signal');td=rt.load(seed,target,'D_signal','signal')
    hs,ht=rt.halves(sd),rt.halves(td)
    reference=rt.REPORT/'sources'/('s%d_t%d'%(seed,target));stream=read(reference/'STREAM.json')
    algo=rt.build(seed,target,run);begin=time.monotonic()
    try:
        init=read(reference/'INITIALIZATION.json')
        assert g.digest(algo.optimizer.state_dict())==init['optimizer_initial_sha256']
        assert g.digest(algo.scheduler.state_dict())==init['scheduler_initial_sha256']
        rt.load_model(algo,rt.model_path(seed,'source',source));position=0;effective=0;done=[]
        recovery=rt.DATA/'active_target_recovery.pt'
        if recovery.exists():
            saved=torch.load(recovery,map_location='cpu')
            if saved['unit']==sid:
                assert saved['seal_sha256']==sha(signal/'C0_SEALED.json') and saved['stream_sha256']==sha(reference/'STREAM.json')
                g.restore(algo,saved['state']);assert all(g.compare_states(saved['state'],g.capture(algo)).values())
                position=saved['position'];effective=saved['effective'];done=saved['done']
            del saved
        def save_state(call):
            env.atomic_save(recovery,dict(unit=sid,position=call,effective=effective,done=done,state=g.capture(algo),seal_sha256=sha(signal/'C0_SEALED.json'),stream_sha256=sha(reference/'STREAM.json')))
        for k in range(position,300):
            if shutil.disk_usage(rt.DATA).free<2*1024**3:raise RuntimeError('STORAGE_RESERVE')
            algo.policy.train();env.seed_all(stream['policy_seed'][k]);lr=algo.optimizer.param_groups[0]['lr'];assert lr==stream['learning_rates'][k]
            before=tuple(p.detach().clone() for p in algo.policy.parameters() if p.requires_grad) if k<50 else None
            loss=float(g.native(algo,rt.batch(data,stream['batches'][k]['indices']),target,stream['batches'][k]['epoch']))
            assert math.isfinite(loss) and algo._optimizer_step_count==algo._scheduler_step_count==k+1
            dn=None
            if before is not None:
                dn=g.norm(tuple(p.detach()-v for p,v in zip((p for p in algo.policy.parameters() if p.requires_grad),before)))
                if lr>0 and dn>0:effective+=1
                del before
            direct=read(reference/'steps'/('%04d.json'%(k+1)))
            assert direct['record_indices']==stream['batches'][k]['indices'] and direct['lr']==lr and direct['lr_after']==algo.optimizer.param_groups[0]['lr']
            write(report/'steps'/('%04d.json'%(k+1)),dict(call=k+1,lr=lr,lr_after=algo.optimizer.param_groups[0]['lr'],record_indices=stream['batches'][k]['indices'],policy_seed=stream['policy_seed'][k],loss=loss,d_norm=dn,effective_prefix=effective,optimizer_step=k+1,scheduler_step=k+1))
            if k<50 and effective in (1,5,20) and effective not in done:
                q=effective;witness=g.capture(algo);now_old=extract(algo,sd)['nll'];now_new=extract(algo,td)['nll']
                assert all(g.compare_states(witness,g.capture(algo)).values())
                ref=read(reference/('D_PREFIX_%04d.json'%(k+1)));base_after=np.array(ref['nll'])
                halves=[]
                for h in (0,1):
                    si=hs==h;ti=ht==h;old=now_old-start_old;gt=start_new-now_new;gb=base_new-base_after
                    hh=dict(c0['halves'][h]);hh.update({'base_progress%d'%q:avg(gb[ti],ids(td)[ti]),'relative_progress%d'%q:avg((gt-gb)[ti],ids(td)[ti]),'source_damage%d'%q:avg(old[si],ids(sd)[si]),'source_tail%d'%q:tail(old[si],ids(sd)[si])})
                    if q==20:
                        five=read(signal/'C5.json')['halves'][h];hh['damage_5_to_20']=hh['source_damage20']-five['source_damage5']
                    halves.append(hh)
                scores={f:mean_optional([h[f] for h in halves]) for f in halves[0]}
                result=dict(seed=seed,source=source,target=target,call=k+1,effective=q,reference_effective=ref['effective'],same_call_comparison=True,scores=scores,halves=halves,predictions=predict(scores,q),state_sha256={f:g.digest(v) for f,v in witness.items()},source_old_nll=now_old.tolist(),source_new_nll=now_new.tolist(),base_target_nll=base_after.tolist(),formal_training_resumed=False)
                path=signal/('C%d.json'%q)
                if path.exists():
                    old=read(path);assert old==result,'PREFIX_RESUME_MISMATCH'
                else:write(path,result)
                sealpath=signal/('C%d_SEALED.json'%q)
                if not sealpath.exists():freeze(sealpath,dict(signals_sha256=sha(path),finalists_sha256=sha(OUT/'FINALISTS_FROZEN.json'),sealed_unix=time.time(),target_calls_completed=k+1,formal_training_resumed=False))
                else:assert read(sealpath)['signals_sha256']==sha(path)
                done.append(q);save_state(k+1);del witness
            if k+1==50:
                write(signal/'PREFIX_AVAILABILITY.json',dict(available=done,unavailable=[q for q in (1,5,20) if q not in done],max_calls=50))
            if k+1 in (100,300):
                path=run/('step%04d.pt'%(k+1));env.atomic_save(path,dict(model=g.cpu_state(g.module(algo).state_dict()),seed=seed,source=source,target=target,call=k+1))
                write(report/('MODEL_%04d.json'%(k+1)),dict(path=str(path),sha256=sha(path)));save_state(k+1)
            if (k+1)%20==0:print('TARGET',sid,k+1,round(time.monotonic()-begin,2),flush=True)
        write(report/'COMPLETE.json',dict(status='PASS',calls=300,seconds=time.monotonic()-begin,seed=seed,source=source,target=target,probe_available=done,stream_sha256=sha(reference/'STREAM.json'),C0_seal_sha256=sha(signal/'C0_SEALED.json'),model_sha256=sha(run/'step0300.pt'),formal_training_resumed=False,diagnostic_training_performed=True))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('seed','source','target'):p.add_argument('--'+key,type=int,required=True)
    a=p.parse_args();access=rt.data_firewall()
    try:main(a.seed,a.source,a.target)
    finally:write(rt.REPORT/'access'/('target_'+unit(a.seed,a.source,a.target)+'.json'),dict(paths=sorted(access),formal_training_resumed=False))
