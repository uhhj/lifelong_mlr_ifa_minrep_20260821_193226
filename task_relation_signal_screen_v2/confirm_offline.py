"""Held-out E_eval reporting only, after prospective prediction seals."""
import argparse
import time
from .common import *
from . import confirm_runtime as rt

def main(seed,source,target):
    import torch
    from task_relation_forecast_v1.runtime import measure
    sid=unit(seed,source,target);dest=rt.REPORT/'offline'/sid
    if (dest/'COMPLETE.json').exists():return
    assert read(rt.REPORT/'transfers'/sid/'COMPLETE.json')['status']=='PASS'
    signal=rt.REPORT/'signals'/sid
    for q in (0,1,5,20):
        path=signal/('C%d_SEALED.json'%q)
        if path.exists():assert read(path)['signals_sha256']==sha(signal/('C%d.json'%q))
    old=rt.load(seed,source,'E_eval','evaluate');new=rt.load(seed,target,'E_eval','evaluate')
    algo=rt.build(seed,target,rt.DATA/'offline_bootstrap'/sid);begin=time.monotonic();result=[]
    try:
        for kind,src,tgt,step in [('base',None,None,0),('source',source,None,1000),('reference',target,None,100),('reference',target,None,300),('transfer',source,target,100),('transfer',source,target,300)]:
            path=rt.model_path(seed,kind,src,tgt,step);rt.load_model(algo,path)
            for d in (old,new):result.append(dict(kind=kind,step=step,task=d['task'],checkpoint_sha256=sha(path),**measure(algo,d)))
        write(dest/'NLL.json',dict(seed=seed,source=source,target=target,results=result,seconds=time.monotonic()-begin,formal_training_resumed=False))
        write(dest/'COMPLETE.json',dict(status='PASS',nll_sha256=sha(dest/'NLL.json'),formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('seed','source','target'):p.add_argument('--'+k,type=int,required=True)
    a=p.parse_args();main(a.seed,a.source,a.target)
