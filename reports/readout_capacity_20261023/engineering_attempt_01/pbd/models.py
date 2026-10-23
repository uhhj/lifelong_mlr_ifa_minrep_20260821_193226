"""Fusion controls and matched global/patch three-entity queries, no patch mean."""
import torch
from torch import nn
from .common import OLD
from .readout_base import LightReadout

class FusionReadout(LightReadout):
 def __init__(self,kind,temporal_ffn=128):
  super().__init__('P3',temporal_ffn=temporal_ffn);self.kind=kind
  if kind=='C':self.fuse=nn.Linear(4*128,128)
  else:
   self.entity_queries=nn.Parameter(torch.randn(3,128)*.02)
   self.attention=nn.MultiheadAttention(128,4,dropout=0,batch_first=True)
   self.fuse=nn.Linear(3*128,128)
 def context(self,x):
  b,t,m,e=x.shape
  h=self.proj(x)+self.modality[None,None]
  h=self.frame(h.reshape(b*t,m,128))
  if self.kind=='C':h=self.fuse(h.flatten(1))
  else:
   q=self.entity_queries[None].expand(b*t,-1,-1)
   h=self.fuse(self.attention(q,h,h,need_weights=False)[0].flatten(1))
  h=h.reshape(b,t,128)+self.time[:t]
  causal=torch.triu(torch.full((t,t),float('-inf'),device=x.device),diagonal=1)
  return self.temporal(h,mask=causal)[:,-1]

class SpatialReadout(FusionReadout):
 def __init__(self,temporal_ffn=128):
  super().__init__('Q',temporal_ffn=temporal_ffn);del self.frame
  # Per-modality spatial positions are fixed, identical in G/S. G repeats its global value.
  ys,xs=torch.meshgrid(torch.linspace(-1,1,14),torch.linspace(-1,1,14),indexing='ij')
  grid=torch.stack([xs,ys],-1).reshape(196,2)
  self.register_buffer('positions',torch.cat([grid.sin(),grid.cos()],-1).repeat(1,32))
 def forward(self,x,patch=None):
  b,t,_,_=x.shape
  visual=x[:,:,1:3,None,:].expand(-1,-1,-1,196,-1) if patch is None else patch
  assert visual.shape==(b,t,2,196,512)
  lang=self.proj(x[:,:,0])+self.modality[0]
  state=self.proj(x[:,:,3])+self.modality[3]
  vision=self.proj(visual)+self.modality[None,None,1:3,None,:]+self.positions[None,None,None]
  # Entity cross-attention before temporal modeling; all 392 visual tokens remain available.
  h=torch.cat([lang[:,:,None],vision.flatten(2,3),state[:,:,None]],2).reshape(b*t,394,128)
  q=self.entity_queries[None].expand(b*t,-1,-1)
  h=self.fuse(self.attention(q,h,h,need_weights=False)[0].flatten(1)).reshape(b,t,128)+self.time[:t]
  causal=torch.triu(torch.full((t,t),float('-inf'),device=x.device),diagonal=1)
  return self.head(self.temporal(h,mask=causal)[:,-1])

def build(kind,temporal_ffn=128):
 if kind in ('P0','M'):return LightReadout('P0' if kind=='P0' else 'P3',temporal_ffn=temporal_ffn)
 if kind in ('C','Q'):return FusionReadout(kind,temporal_ffn=temporal_ffn)
 if kind in ('G','S'):return SpatialReadout(temporal_ffn=temporal_ffn)
 raise ValueError(kind)
