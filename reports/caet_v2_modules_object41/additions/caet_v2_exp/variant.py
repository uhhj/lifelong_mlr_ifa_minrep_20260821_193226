"""Variant identity is bound by the immutable per-worker ENTRY.json."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
ENTRY=Path(__file__).resolve().parents[2]/'ENTRY.json'
VARIANT=json.loads(ENTRY.read_text())['variant']
assert VARIANT in ('TAC','RSET','EFTR')
METHOD='CAET-v2+'+VARIANT
def feature(x):
    if x.ndim==5:x=x.squeeze(1)
    assert x.shape[1:]==(8,4,512)
    return x[:,:7] if VARIANT=='TAC' else x
def exec_params(policy,x):
    if x.ndim==5:x=x.squeeze(1)
    z=policy.temporal_encode(x)[:,-1]
    means,std,logits=policy.policy_head.forward_fn(z)
    mode=logits.argmax(-1);ix=torch.arange(len(mode),device=mode.device)
    return means[ix,mode],std[ix,mode],mode
def exec_kl(ref,now):
    m0,s0,_=ref;m,s,_=now
    return (torch.log(s/s0)+(s0.square()+(m0-m).square())/(2*s.square())-.5).sum(-1)
def phase(row):
    g=row['group'].split('/')
    if len(g)!=4:return 'INVALID'
    return ('STATIC' if g[1]=='STATIC' else 'TRANSLATION' if g[1].startswith('T') else 'ROTATION')+'/'+g[-1]
def descriptor(x):
    if x.ndim==5:x=x.squeeze(1)
    return F.normalize(x[:,-1,1:,:].reshape(len(x),1536),dim=-1)

class ActionSupport:
    def __init__(self,ds,device):
        ids=[i for i,r in enumerate(ds.rows) if r['valid'] and not r['holdout']]
        self.ids=ids
        self.desc=torch.cat([descriptor(torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ids[j:j+64]])) for j in range(0,len(ids),64)]).to(device)
        self.actions=torch.stack([ds.rows[i]['next_action'] for i in ids]).to(device)
        self.scale=self.actions.std(0,unbiased=False).clamp_min(.05)
        demos=sorted({(ds.rows[i]['task'],ds.rows[i]['demo']) for i in ids})
        self.demo=torch.tensor([demos.index((ds.rows[i]['task'],ds.rows[i]['demo'])) for i in ids],device=device)
        self.keys=[(ds.rows[i]['task'],ds.rows[i]['demo']) for i in ids]
    @torch.no_grad()
    def mask(self,x,a,exclude_demo=None):
        sim=descriptor(x)@self.desc.T
        if exclude_demo is not None:
            for j,key in enumerate(exclude_demo):
                bad=torch.tensor([k==key for k in self.keys],device=sim.device)
                sim[j,bad]=-2
        values,idx=sim.topk(min(20,len(self.ids)),dim=-1)
        distances=(((a[:,None]-self.actions[idx])/self.scale).square().mean(-1)).sqrt()
        good=(values>=.90)&(distances<=.50)
        return torch.tensor([len(torch.unique(self.demo[idx[j][good[j]]]))>=3 for j in range(len(x))],device=x.device)
