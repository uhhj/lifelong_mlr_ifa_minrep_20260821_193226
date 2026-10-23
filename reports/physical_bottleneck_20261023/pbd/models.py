import torch
from torch import nn
from .common import OLD
from physical_effect_probe.readout_models import LightReadout

class QueryReadout(LightReadout):
 def __init__(self):
  super().__init__('P3');self.query=nn.Parameter(torch.zeros(128))
 def context(self,x):
  b,t,m,e=x.shape;h=self.proj(x)+self.modality[None,None];h=self.frame(h.reshape(b*t,m,128))
  w=(h*self.query).sum(-1).div(128**.5).softmax(-1);h=(h*w[...,None]).sum(1).reshape(b,t,128)+self.time[:t]
  mask=torch.triu(torch.full((t,t),float('-inf'),device=x.device),diagonal=1)
  return self.temporal(h,mask=mask)[:,-1]

class SpatialReadout(nn.Module):
 def __init__(self):
  super().__init__();self.base=LightReadout('P3')
  self.vproj=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128));self.pos=nn.Linear(2,128,bias=False)
  self.query=nn.Parameter(torch.zeros(128));self.vout=nn.Linear(128,512)
  ys,xs=torch.meshgrid(torch.linspace(-1,1,14),torch.linspace(-1,1,14),indexing='ij');self.register_buffer('grid',torch.stack([xs,ys],-1).reshape(196,2))
 def forward(self,x,patch=None):
  # x: B,T,L/agent/wrist/state,E. The paired G/S forward differs only in visual values.
  visual=x[:,:,1:3,None,:].expand(-1,-1,-1,196,-1) if patch is None else patch
  assert visual.shape==(*x.shape[:2],2,196,512)
  h=self.vproj(visual)+self.pos(self.grid);w=(h*self.query).sum(-1).div(128**.5).softmax(-1)
  y=x.clone();y[:,:,1:3]=self.vout((h*w[...,None]).sum(-2));return self.base(y)

def build(kind):
 if kind=='P0':return LightReadout('P0')
 if kind=='M':return LightReadout('P3')
 if kind=='Q':return QueryReadout()
 if kind in ('G','S'):return SpatialReadout()
 raise ValueError(kind)
