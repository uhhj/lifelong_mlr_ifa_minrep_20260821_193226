"""Unmodified C expert path plus separate executed-distribution retention."""
import torch
import torch.nn.functional as F
from legacy_losses import cost,w2,route

def expert(q,y,scale):
    mu,sd,z=q;ix=torch.arange(len(mu),device=mu.device);k=z.detach().argmax(-1)
    c=cost(mu,sd,y.detach(),scale.detach())
    rr=route(z,F.one_hot(c.detach().argmin(-1),z.shape[-1]).bool())
    return (c[ix,k]+.1*rr).mean(),c[ix,k].mean(),rr.mean()

def retain(q,tm,ts,scale):
    mu,sd,z=q;ix=torch.arange(len(mu),device=mu.device);k=z.detach().argmax(-1)
    d=w2(mu,sd,tm.detach(),ts.detach(),scale.detach())
    rr=route(z,F.one_hot(d.detach().argmin(-1),z.shape[-1]).bool())
    return d[ix,k],rr
