import torch
from torch import nn

class Head(nn.Module):
 def __init__(self,dim):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),nn.Linear(128,7))
 def forward(self,x):return self.net(x)

class LightReadout(nn.Module):
 def __init__(self,kind='P3',temporal_ffn=128):
  super().__init__();self.kind=kind;self.proj=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128));self.modality=nn.Parameter(torch.zeros(4,128));self.time=nn.Parameter(torch.zeros(8,128))
  nn.init.normal_(self.modality,std=.02);nn.init.normal_(self.time,std=.02)
  self.frame=nn.TransformerEncoderLayer(128,4,512,dropout=0,batch_first=True)
  self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,temporal_ffn,dropout=0,batch_first=True),2)
  self.head=Head(128)
 def context(self,x):
  assert 1<=x.shape[1]<=8 and x.shape[2:]==(4,512)
  if self.kind=='P4' and x.shape[1]>1:
   x=x.clone();x[:,-1,1:]=x[:,-2,1:]
  if self.kind=='P0':x=x[:,:,[0,3]];mod=self.modality[[0,3]]
  else:mod=self.modality
  b,t,m,e=x.shape;h=self.proj(x)+mod[None,None];h=self.frame(h.reshape(b*t,m,128)).mean(1).reshape(b,t,128)+self.time[:t]
  mask=torch.triu(torch.full((t,t),float('-inf'),device=x.device),diagonal=1)
  return self.temporal(h,mask=mask)[:,-1]
 def forward(self,x):return self.head(self.context(x))

def build(kind):return Head(512) if kind in ('P1','P2') else LightReadout(kind)

def perturb(x,mode,donor=None):
 y=x.clone()
 if mode=='latest_visual_previous':y[:,-1,1:3]=x[:,-2,1:3]
 elif mode=='other_demo_visual':y[:,:,1:3]=donor[:,:,1:3]
 elif mode!='normal':raise ValueError(mode)
 return y
