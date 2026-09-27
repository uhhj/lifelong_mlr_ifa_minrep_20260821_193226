"""Prospective final-candidate signals; no endpoint or evaluation access."""
import argparse
import time
import numpy as np
from .common import *
from . import confirm_runtime as rt
from .measure import extract,ids,mean_optional
from .numeric import avg,mmd,Transform

def predict(scores,when):
    frozen=read(OUT/'FINALISTS_FROZEN.json');rows=[]
    models=[dict(m,model=m['package'],kind='candidate') for m in frozen['candidates']]
    models += [dict(m,kind='baseline') for m in frozen['baselines']]
    for m in models:
        dynamic='base_progress20' in m['features']
        needed=20 if dynamic else 5 if 'base_progress5' in m['features'] else 1 if 'base_progress1' in m['features'] else 0
        if needed!=when:continue
        assert set(m['features'])<=set(scores),('FEATURE_MISSING',m['model'],set(m['features'])-set(scores))
        fit=m['fit'];tr=Transform(m['features'],{});tr.spec=fit['transform'];z,audit=tr.apply([scores])
        raw=float(((z-np.array(fit['center']))@np.array(fit['coef']))[0]*fit['y_std']+fit['y_mean'])
        rows.append(dict(endpoint=m['endpoint'],model=m['model'],kind=m['kind'],when=when,prediction=float(np.clip(raw,-1,1)),raw_prediction=raw,preprocessing_audit=audit))
    return rows

def static(seed,source,target):
    import torch
    from gradient_relation_experiment import kernel as g
    dest=rt.REPORT/'signals'/unit(seed,source,target);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'C0_SEALED.json').exists():
        seal=read(dest/'C0_SEALED.json');assert sha(dest/'C0.json')==seal['signals_sha256'];return
    assert not (rt.REPORT/'transfers'/unit(seed,source,target)/'COMPLETE.json').exists()
    sd=rt.load(seed,source,'D_signal','signal');td=rt.load(seed,target,'D_signal','signal')
    algo=rt.build(seed,target,rt.DATA/'signal_bootstrap'/unit(seed,source,target));begin=time.monotonic()
    try:
        initial=g.capture(algo)
        rt.load_model(algo,rt.model_path(seed,'base'));bs=extract(algo,sd);bt=extract(algo,td)
        rt.load_model(algo,rt.model_path(seed,'source',source));ss=extract(algo,sd);st=extract(algo,td)
        witness=g.capture(algo);repeat=extract(algo,td)
        assert np.array_equal(st['nll'],repeat['nll']) and all(g.compare_states(witness,g.capture(algo)).values())
        hs,ht=rt.halves(sd),rt.halves(td);values=[{},{}]
        scale=np.array(read(rt.REPORT/'DATA_RULES.json')['tasks'][str(target)]['action_std'])
        action=td['actions'][:,-1,:].numpy().astype(float)
        improvement=((bt['expected']-action)/scale)**2-((st['expected']-action)/scale)**2
        distance=mmd(ss['z'],st['z'],hs,ht)
        for h in (0,1):
            si=hs==h;ti=ht==h
            values[h]=dict(b_target=avg(bt['nll'][ti],ids(td)[ti]),b_source=avg(ss['nll'][si],ids(sd)[si]),b_progress=avg((bs['nll']-ss['nll'])[si],ids(sd)[si]),affinity=avg((bt['nll']-st['nll'])[ti],ids(td)[ti]),action_improvement=avg(improvement[ti,:6].mean(1),ids(td)[ti]),gripper_improvement=avg(improvement[ti,6],ids(td)[ti]),mmd=distance[h])
        scores={f:mean_optional([h[f] for h in values]) for f in values[0]}
        arrays=rt.DATA/'signals'/unit(seed,source,target)/'C0_WINDOWS.npz'
        npz(arrays,base_source=bs['nll'],base_target=bt['nll'],source_source=ss['nll'],source_target=st['nll'])
        predictions=predict(scores,0)
        freeze(dest/'C0.json',dict(seed=seed,source=source,target=target,scores=scores,halves=values,predictions=predictions,window_path=str(arrays),window_sha256=sha(arrays),source_checkpoint_sha256=sha(rt.model_path(seed,'source',source)),seconds=time.monotonic()-begin,source_state_unchanged=True,formal_training_resumed=False))
        freeze(dest/'C0_SEALED.json',dict(signals_sha256=sha(dest/'C0.json'),finalists_sha256=sha(OUT/'FINALISTS_FROZEN.json'),sealed_unix=time.time(),target_calls_completed=0,formal_training_resumed=False))
        g.restore(algo,initial);assert all(g.compare_states(initial,g.capture(algo)).values())
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('seed','source','target'):p.add_argument('--'+key,type=int,required=True)
    a=p.parse_args();access=rt.data_firewall()
    try:static(a.seed,a.source,a.target)
    finally:write(rt.REPORT/'access'/('static_'+unit(a.seed,a.source,a.target)+'.json'),dict(paths=sorted(access),formal_training_resumed=False))
