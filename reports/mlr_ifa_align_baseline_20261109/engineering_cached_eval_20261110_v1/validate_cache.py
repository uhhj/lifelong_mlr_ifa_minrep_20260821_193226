"""Offline admission and reference S-cache action parity; never creates envs."""
import sys,json,time,types,hashlib,copy
from pathlib import Path
import torch,numpy as np
from coreset_exp.model import cfg,data_setup,complete,strict_downstream
from coreset_exp.common import O,rng,restore_rng,sha
from libero.lifelong.utils import control_seed
from cache_policy import encode,cached_get_action,admission
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def main():
    torch.cuda.set_per_process_memory_fraction(.23);torch.backends.cudnn.benchmark=True;control_seed(809110)
    c=cfg('eval');b,_,shape=data_setup(c,False);c.shape_meta=shape;p=complete(c,shape).eval().requires_grad_(False)
    instr,mask=b.get_task_emb(8);results=[];case=O.name
    stages=range(6,10) if case.startswith('OBJECT') else [None]
    for stage in stages:
        if stage is not None:
            cp=O/'random/models'/f'mlrifa_{stage}_model_ep99.pth';strict_downstream(p,cp)
        before=rng();proof=admission(p,shape,instr,mask,rng,restore_rng);assert equal(before,rng())
        gen=torch.Generator().manual_seed(20261024)
        obs={k:torch.rand((2,10,*v),generator=gen).cuda() for k,v in shape['all_shapes'].items()}
        base=dict(task_emb=instr.repeat(2,1).cuda(),masks=mask.repeat(2,1).cuda());p.reset();history=[];rows=[]
        with torch.no_grad():
            for i in range(10):
                d=dict(obs={k:v[:,max(0,i-7):i+1].contiguous() for k,v in obs.items()},**base)
                saved=rng();a=cached_get_action(p,d);after=rng();restore_rng(saved)
                new=dict(obs={k:v[:,i:i+1] for k,v in obs.items()},**base)
                history.append(encode(p,new));history=history[-8:]
                dist=p.policy_head(p.temporal_encode(torch.cat(history,1))[:,-1],train=False)
                ref=dist.sample().detach().cpu().numpy()
                assert np.array_equal(a,ref) and equal(after,rng())
                rows.append(dict(step=i+1,history=min(i+1,8),same_as_reference_cached_action=True,same_rng=True))
        results.append(dict(stage=stage,proof=proof,wrapper_parity=rows,checkpoint_sha256=sha(cp) if stage is not None else sha(c.pretrain)))
    print('RESULT_JSON='+json.dumps(dict(status='PASS_CACHE_TOLERANCE_AND_REFERENCE_WRAPPER',case=case,stages=results,new_optimizer=0,new_environment=0,production_writes=0)))
if __name__=='__main__':main()
