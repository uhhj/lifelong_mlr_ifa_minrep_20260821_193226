import types
from .common import *
from torch import nn
from torch.distributions import Independent,Normal
from execution_core import calibrate_distribution

def temporal(t0,h):
    b,t,m,e=h.shape
    return t0(h.reshape(b,t*m,e)).reshape(b,t,m,e)[:,-1,0]

def raw(head,z):
    assert not head.low_eval_noise
    x=head.share(z)
    return (head.mean_layer(x).view(-1,5,7),head.actv(head.logstd_layer(x).view(-1,5,7))+head.min_std,head.logits_layer(x))

def decode(p,t0,bank,h,ids=None):
    with torch.no_grad():
        z=p.temporal_encode(h)[:,-1];b,s,l=raw(p.policy_head,z);q=temporal(t0,h)
    delta=bank(q,h[:,-1,0],ids)
    out=calibrate_distribution(b,s,l,delta,EPSILON);out.update(delta=delta,q=q,b=b)
    return out

def selected_dist(out):
    return out['executed_dist']

class NativeHead(nn.Module):
    def __init__(self,base,t0,bank,ctx):
        super().__init__();self.base=base;self.t0=t0;self.bank=bank;self.ctx=ctx
    def forward(self,z,train=False):
        h=self.ctx['h'];b,s,l=raw(self.base,z);q=temporal(self.t0,h)
        out=calibrate_distribution(b,s,l,self.bank(q,h[:,-1,0]),EPSILON);self.ctx['out']=out
        self.ctx['max_executed_kl']=max(self.ctx.get('max_executed_kl',0.),float(out['actual_kl'].detach().max()))
        self.ctx['max_raw_kl']=max(self.ctx.get('max_raw_kl',0.),float(out['raw_kl'].detach().max()))
        self.ctx['projected_states']=self.ctx.get('projected_states',0)+int((out['alpha'].detach()<1).sum())
        self.ctx['sampled_states']=self.ctx.get('sampled_states',0)+len(h)
        return selected_dist(out)

def install_native(p,t0,bank):
    ctx={};original=p.spatial_encode;base=p.policy_head
    def spatial(data):
        h=original(data);ctx['h']=h;ctx['spatial_calls']=ctx.get('spatial_calls',0)+1;return h
    p.spatial_encode=spatial;p.policy_head=NativeHead(base,t0,bank,ctx)
    return ctx

def update(p,t0,bank,opt,sch,h,actions,ids):
    opt.zero_grad(set_to_none=True);out=decode(p,t0,bank,h,ids)
    loss=-selected_dist(out).log_prob(actions[:,-1]).mean();assert torch.isfinite(loss);loss.backward()
    present=set(ids.tolist());absent=[]
    for t in bank.task_ids:
        pars=list(bank.branches[str(t)].parameters())
        if t not in present:
            assert all(v.grad is None for v in pars);absent.extend((v,v.detach().clone(),cpu(opt.state.get(v,{}))) for v in pars)
        else:assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in pars)
    assert all(v.grad is None for v in p.parameters()) and all(v.grad is None for v in t0.parameters())
    norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),100.)
    assert torch.isfinite(norm)
    opt.step();sch.step()
    for v,b,s in absent:
        assert torch.equal(v,b);now=opt.state.get(v,{})
        assert set(now)==set(s)
        for key in s:assert torch.equal(now[key].cpu(),s[key]) if torch.is_tensor(s[key]) else now[key]==s[key]
    return dict(loss=float(loss.detach()),grad_norm=float(norm),delta_max=float(out['delta'].abs().max()),active=sorted(present),
                actual_kl_max=float(out['actual_kl'].detach().max()),raw_kl_max=float(out['raw_kl'].detach().max()),
                projected_states=int((out['alpha'].detach()<1).sum()),states=len(h),rounding_passes=out['rounding_passes'])
