import json,types,time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal,Independent,Categorical,MixtureSameFamily
from easydict import EasyDict
from .common import O,PROTO,digest,rng_digest
from coreset_exp.model import angular_mean,strict_downstream,data_setup,complete

class Channel(nn.Module):
    def __init__(self,w,task):
        super().__init__();self.register_buffer('gate_w',w.detach().clone().double())
        gen=torch.Generator(device='cpu').manual_seed(42+task)
        self.V=nn.Parameter(torch.randn(8,512,generator=gen)*.02)
        self.U=nn.Parameter(torch.zeros(75,8))
    def gate(self,e):
        assert self.gate_w.dtype==torch.float64
        e=e.detach().double();return(e/e.norm(dim=-1,keepdim=True))@self.gate_w
    def residual(self,z):return(z@self.V.T)@self.U.T
    def forward(self,z,e):return(self.gate(e)[:,None]*self.residual(z).double()).to(z.dtype)
def raw_head(head,z):
    x=head.share(z);return torch.cat([head.mean_layer(x),head.logstd_layer(x),head.logits_layer(x)],-1)
def decode(head,raw):
    means=raw[:,:35].reshape(-1,5,7).tanh();scales=head.actv(raw[:,35:70].reshape(-1,5,7))+head.min_std
    logits=raw[:,70:];selected=logits.argmax(-1)
    return dict(raw=raw,means=means,scales=scales,logits=logits,probs=logits.softmax(-1),selected=selected,gmm=MixtureSameFamily(Categorical(logits=logits),Independent(Normal(means,scales),1)))
def readout(p,z,e,disabled=()):
    raw=raw_head(p.policy_head,z)
    for n,ch in p.residual_bank.items():
        if n not in disabled:raw=raw+ch(z,e)
    return decode(p.policy_head,raw)
def forward(p,h):
    z=p.temporal_encode(h)[:,-1];return readout(p,z,h[:,-1,0])['gmm'],z
def get_action(p,data):
    started=time.perf_counter()
    p.eval()
    with torch.no_grad():
        data=p.preprocess_input(data,train_mode=False);h=p.spatial_encode(data);z=p.temporal_encode(h)[:,-1]
        out=readout(p,z,h[:,-1,0]);ix=torch.arange(len(z),device=z.device);j=out['selected']
        action=Independent(Normal(out['means'][ix,j],out['scales'][ix,j]),1).sample().detach().cpu().reshape(-1,7).numpy()
        if hasattr(p,'action_latency'):p.action_latency.append(time.perf_counter()-started)
        return action
def attach(p,online=False):
    assert not hasattr(p,'residual_bank');p.residual_bank=nn.ModuleDict()
    if online:p.get_action=types.MethodType(get_action,p);p.action_latency=[]
    else:p.forward=types.MethodType(forward,p)
    return p
def register(p,t):
    before=digest(p.state_dict());r=rng_digest()
    assert list(p.residual_bank)==[str(i) for i in range(6,t)]
    gates=np.load(PROTO/'geometry/gates.npz');ch=Channel(torch.from_numpy(gates['gates'][t-6]),t).to(next(p.parameters()).device)
    p.residual_bank[str(t)]=ch
    for n,v in p.named_parameters():
        allowed=n.startswith(('temporal_transformer.','policy_head.')) or n in (f'residual_bank.{t}.U',f'residual_bank.{t}.V')
        v.requires_grad_(allowed)
        if not allowed:v.grad=None
    after=digest(p.state_dict())
    assert all(after[n]==v for n,v in before.items()) and rng_digest()==r
    return dict(task=t,global_rng_unchanged=True,existing_tensors_unchanged=True,new_tensors=digest(ch.state_dict()),protected=list(range(t)),trainable=[n for n,v in p.named_parameters() if v.requires_grad])
def cfg(name):
    from .common import SEED
    c=EasyDict(json.loads((O/f'resolved_{name}.json').read_text()))
    assert c.seed==SEED and all(c.policy[k] is True for k in ('use_state','use_language','use_vision'))
    return c
def load_stage(p,path,stage):
    raw=torch.load(path,map_location='cpu');assert raw['task']==stage and raw['epoch_index']==99
    assert raw['branches']==list(range(6,stage+1))
    attach(p,online=hasattr(p,'spatial_encode'))
    for t in range(6,stage+1):register(p,t)
    sd={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in raw['state_dict'].items()}
    want={k for k in p.state_dict() if k.startswith(('temporal_transformer.','policy_head.','residual_bank.'))}
    assert set(sd)==want
    target=p.state_dict()
    for k,v in sd.items():assert v.shape==target[k].shape and v.dtype==target[k].dtype,k
    result=p.load_state_dict(sd,strict=False);assert not result.unexpected_keys and not(set(result.missing_keys)&want)
    assert digest({k:v for k,v in p.state_dict().items() if k in want})==digest(sd)
    gates=np.load(PROTO/'geometry/gates.npz')['gates']
    for n,ch in p.residual_bank.items():assert torch.equal(ch.gate_w.cpu(),torch.from_numpy(gates[int(n)-6]))
    p.requires_grad_(False)
    return len(want)
