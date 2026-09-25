"""Independent E_eval reader; never exports outcomes to the signal producer."""
import argparse
from .common import OUT,STORE,read,write,sha
from . import runtime as rt


def main(seed,source,target):
    import torch
    sid='s%d_%d_%d'%(seed,source,target);dest=OUT/'offline'/sid
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    assert (OUT/'signals'/sid/'SEALED.json').exists()
    assert read(OUT/'transfers'/sid/'COMPLETE.json')['status']=='PASS'
    algo=rt.build(seed,target,STORE/'evaluation_bootstrap'/sid)
    try:
        datasets={t:rt.load_data(seed,t,'E_eval','evaluate') for t in (source,target)}
        specs=[('base',rt.model_path(seed,'base'),0)]
        specs += [('source%d'%source,rt.model_path(seed,'source',source=source,step=k),k) for k in (100,300,1000)]
        specs += [('reference%d'%target,rt.model_path(seed,'source',source=target,step=k),k) for k in (100,300)]
        specs += [('transfer%d_%d'%(source,target),rt.model_path(seed,'transfer',source=source,target=target,step=k),k) for k in (100,300)]
        rows=[];summaries=[]
        for model,path,k in specs:
            rt.load_model(algo,path)
            for t,data in datasets.items():
                values=rt.measure(algo,data)
                metadata=dict(seed=seed,source=source,target=target,model=model,step=k,eval_task=t)
                rows.extend(dict(metadata,**r) for r in values['rows'])
                summaries.append(dict(metadata,mean_trajectory_nll=values['mean_nll'],model_sha256=sha(path)))
        rt.table(dest/'WINDOW_NLL.csv',rows);rt.table(dest/'OFFLINE_NLL.csv',summaries)
        write(dest/'COMPLETE.json',dict(status='PASS',role='E_eval',signal_seal_sha256=sha(OUT/'signals'/sid/'SEALED.json'),
              windows=len(rows),formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('seed','source','target'):p.add_argument('--'+k,type=int,required=True)
    a=p.parse_args();main(a.seed,a.source,a.target)
