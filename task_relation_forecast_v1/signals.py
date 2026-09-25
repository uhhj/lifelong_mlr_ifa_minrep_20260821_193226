"""Seal prospective signals before any target-branch adaptation."""
import argparse
import copy
import gc
import time
import numpy as np
from .common import ROOT,OUT,STORE,read,write,freeze,sha,stable_seed
from . import runtime as rt


def latent_centroid(data):
    values={}
    for h,row in zip(data['feature'],data['rows']):
        q=h[:row['valid_frame_count']].double().mean(0).flatten()
        q=q/q.norm().clamp_min(1e-30)
        values.setdefault(row['trajectory_id'],[]).append(q)
    import torch
    return torch.stack([torch.stack(v).mean(0) for v in values.values()]).mean(0)


def gn_basis(algo,data,weights,config):
    import torch
    from scipy.sparse.linalg import LinearOperator,eigsh
    from gn_functional_geometry_v1.engine import Engine
    from gradient_relation_experiment import kernel as g
    start=time.monotonic();z=rt.terminal(algo,data)
    e=Engine(copy.deepcopy(g.module(algo).policy_head).double(),{'old_build':z.double()},
             {'old_build':data['actions'][:,-1,:].cuda().double()},torch.tensor(weights,device='cuda',dtype=torch.float64))
    trace=e.trace()
    if trace==0:
        return np.zeros((e.theta.numel(),0)),dict(rank=0,trace=0.,status='ZERO_TRACE_UNAVAILABLE',seconds=time.monotonic()-start)
    assert trace>0 and np.isfinite(trace)
    op=LinearOperator((e.theta.numel(),e.theta.numel()),matvec=lambda v:e.cv(v).cpu().numpy(),dtype=np.float64)
    vals,u=eigsh(op,k=config['rank'],ncv=config['ncv'],tol=config['tol'],maxiter=config['maxiter'],
                 v0=np.random.RandomState(config['seed']).normal(size=e.theta.numel()))
    idx=np.argsort(vals)[::-1];vals=vals[idx];u=u[:,idx].copy()
    gramerr=float(np.max(abs(u.T@u-np.eye(len(vals)))))
    rank=int(np.sum(vals>max(1e-12,vals[0]*1e-10)))
    residuals=[float(np.linalg.norm(op@u[:,k]-vals[k]*u[:,k]))/max(abs(vals[k]),1e-30) for k in range(rank)]
    assert gramerr<=2e-5 and (not residuals or max(residuals)<=.002)
    assert vals.min()>=-1e-10*vals.max() and vals.sum()<=trace*(1+1e-5)
    info=dict(rank=rank,eigenvalues=vals.tolist(),trace=trace,orthogonal_error=gramerr,
              relative_eigen_residuals=residuals,parameter_names=e.names,parameter_sizes=e.sizes,
              jvp_count=e.jvps,vjp_count=e.vjps,seconds=time.monotonic()-start,
              status='PASS',basis_retention='transient; frozen operator, data, seeds and spectrum retained')
    del e,z;gc.collect();torch.cuda.empty_cache()
    return u[:,:rank],info


def main(seed,source,target):
    import torch
    from gradient_relation_experiment import kernel as g
    from cb_ifa_experiment import gpu_runtime as env
    from libero.lifelong.algos.mlr import _pairwise_cosine_mean
    sid='s%d_%d_%d'%(seed,source,target);dest=OUT/'signals'/sid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'SEALED.json').exists():return
    conf=read(OUT/'SIGNAL_IMPLEMENTATION_PROTOCOL.json')
    assert not (OUT/'transfers'/sid/'COMPLETE.json').exists(), 'Signals must precede target adaptation'
    begin=time.monotonic();algo=rt.build(seed,target,STORE/'signal_bootstrap'/sid)
    try:
        sd=rt.load_data(seed,source,'D_signal','signal',source,target)
        td=rt.load_data(seed,target,'D_signal','signal',source,target)
        assert sd['encoder_sha256']==td['encoder_sha256']
        src=rt.model_path(seed,'source',source=source,step=1000)
        base=rt.model_path(seed,'base')
        rt.load_model(algo,base);bt=rt.measure(algo,td)
        rt.load_model(algo,src);bs=rt.measure(algo,sd);st=rt.measure(algo,td)
        sv=sd['feature'][:,:,1,:].reshape(-1,512);tv=td['feature'][:,:,1,:].reshape(-1,512)
        sl=sd['feature'][:,0,0,:];tl=td['feature'][:,0,0,:]
        cs,ct=latent_centroid(sd),latent_centroid(td)
        scores=dict(seed=seed,source=source,target=target,b_t=bt['mean_nll'],b_s=bs['mean_nll'],
                    A=bt['mean_nll']-st['mean_nll'],S_visual=_pairwise_cosine_mean(sv,tv),
                    S_language=_pairwise_cosine_mean(sl,tl),S_lat=float(cs@ct/(cs.norm()*ct.norm())))
        write(dest/'NLL_SIGNALS.json',dict(base_on_target=bt,source_on_source=bs,source_on_target=st))
        a=sd['actions'][:,-1,:].double();weights=[]
        for lo,hi in ((0,3),(3,6),(6,7)):
            scale=max(.05,float(a[:,lo:hi].var(0,unbiased=False).mean().sqrt()))
            weights.extend([1/((hi-lo)*scale**2)]*(hi-lo))
        write(OUT/'STATUS.json',dict(stage='PILOT_RELATION_GEOMETRY',unit=sid,
              diagnostic_training_performed=True,formal_training_resumed=False))
        u,si=gn_basis(algo,sd,weights,conf['GN']);v,ti=gn_basis(algo,td,weights,conf['GN'])
        assert si.get('parameter_names')==ti.get('parameter_names') or min(si['rank'],ti['rank'])==0
        r=min(si['rank'],ti['rank'])
        scores['S_GN']=float(np.sum((u.T@v)**2)/r) if r else None
        write(dest/'GN_GEOMETRY.json',dict(source=si,target=ti,weights=weights,
              same_parameter_point_sha256=g.digest(algo.policy.state_dict()),subspace_overlap=scores['S_GN']))
        del u,v;gc.collect()
        # E_eval is never loaded in this process. Only target A first batch is
        # available to the expensive probe; source A is not accessible here.
        tdtrain=rt.load_data(seed,target,'A_train','shadow',source,target)
        stream=read(OUT/'sources'/('s%d_t%d'%(seed,target))/'STREAM.json')
        algo.policy.train();before=g.capture(algo);shadow_start=time.monotonic()
        env.seed_all(stream['policy_seed'][0])
        g.native(algo,rt.batch(tdtrain,stream['batches'][0]['indices']),target,0)
        after=g.capture(algo);witness={k:g.digest(v) for k,v in after.items()}
        so=rt.measure(algo,sd);sn=rt.measure(algo,td)
        scores.update(R_old=so['mean_nll']-bs['mean_nll'],R_new=sn['mean_nll']-st['mean_nll'])
        g.restore(algo,before);checks=g.compare_states(before,g.capture(algo));assert all(checks.values())
        shadow=dict(rollback=checks,after_witness=witness,first_batch=stream['batches'][0],
                    initial_state_witness={k:g.digest(v) for k,v in before.items()},
                    lr=stream['learning_rates'][0],seconds=time.monotonic()-shadow_start,
                    target_stream_sha256=sha(OUT/'sources'/('s%d_t%d'%(seed,target))/'STREAM.json'))
        write(dest/'SHADOW_ROLLBACK.json',shadow)
        write(dest/'SIGNALS.json',dict(scores=scores,source_checkpoint=str(src),source_sha256=sha(src),
              base_checkpoint=str(base),base_sha256=sha(base),encoder_sha256=sd['encoder_sha256'],
              seconds=time.monotonic()-begin,signal_data_roles=['source D_signal','target D_signal','target A first batch'],
              E_eval_accessed=False,source_A_accessed=False))
        freeze(dest/'SEALED.json',dict(signals_sha256=sha(dest/'SIGNALS.json'),shadow_sha256=sha(dest/'SHADOW_ROLLBACK.json'),
              geometry_sha256=sha(dest/'GN_GEOMETRY.json'),sealed_unix_time=time.time(),target_steps_committed=0,
              formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('seed','source','target'):p.add_argument('--'+k,type=int,required=True)
    a=p.parse_args();main(a.seed,a.source,a.target)
