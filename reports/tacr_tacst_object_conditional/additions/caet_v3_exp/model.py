import torch,math
from torch import nn
from torch.nn import functional as F
class DeployResidual(nn.Module):
 def __init__(self):
  super().__init__()
  self.R=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,128),nn.LayerNorm(128))
  self.U=nn.Linear(128,512)
  self.gate=nn.Sequential(nn.Linear(640,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
  nn.init.zeros_(self.U.weight);nn.init.zeros_(self.U.bias)
  nn.init.zeros_(self.gate[2].weight);nn.init.constant_(self.gate[2].bias,math.log(.1/.9))
 def forward(self,h):
  frozen=h.detach();c=self.R(frozen);gate=self.gate(torch.cat([frozen,c.detach()],-1))
  rms=frozen.square().mean(-1,keepdim=True).clamp_min(1e-16).sqrt()
  delta=.2*rms*gate*torch.tanh(self.U(c))
  return h+delta,c,gate,delta
class Student(nn.Module):
 def __init__(self):
  super().__init__();self.deploy=DeployResidual();self.E=nn.Sequential(nn.Linear(135,128),nn.ReLU(),nn.Linear(128,128));self.D=nn.Linear(128,7)
 def condition(self,c,a):return self.E(torch.cat([c,a],-1))
 def representation_parameters(self):return list(self.deploy.R.parameters())+list(self.E.parameters())+list(self.D.parameters())
 def control_parameters(self):return list(self.deploy.U.parameters())+list(self.deploy.gate.parameters())
def feature(x):
 if x.ndim==5:x=x.squeeze(1)
 assert x.shape[1:]==(8,4,512);return x
def hidden(p,x):return p.temporal_encode(feature(x))[:,-1]
def masked_bc(policy,h,x,aligned=True):
 gmm=policy.policy_head(h);mask=x['causal_bc_valid'].bool();v=-gmm.log_prob(x['next_action'])*policy.policy_head.loss_coef
 return v[mask].mean() if mask.any() else v.sum()*0,mask
class ResidualHead(nn.Module):
 """CompletePolicy deployment API with a strictly loaded residual."""
 def __init__(self,head,branch):super().__init__();self.base=head;self.branch=branch
 def forward(self,h,train=True):return self.base(self.branch(h)[0],train=train)
 def forward_fn(self,h):return self.base.forward_fn(self.branch(h)[0])
 @property
 def loss_coef(self):return self.base.loss_coef
