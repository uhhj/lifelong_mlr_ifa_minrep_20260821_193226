"""Training-only next-action head; never registered under the deployed policy."""
import copy
import torch
from coreset_exp.common import O,SEED,rng,restore_rng
from .variant import feature

class AuxiliaryHeadMixin:
 def init_aux(self,t):
  # deepcopy consumes no RNG; formal policy structure and optimizer are untouched.
  self.aux_head=copy.deepcopy(self.policy.module.policy_head).cuda()
  self.aux_optimizer=torch.optim.AdamW(self.aux_head.parameters(),lr=.0001,weight_decay=.01)
  self.aux_steps=0
  if t>6:
   raw=torch.load(O/'ra_coreset/aux_heads'/('task%d_ep99.pth'%(t-1)),map_location='cpu')
   assert raw['task']==t-1 and raw['epoch']==99
   self.import_aux_state(raw['state'])
  self.aux_head.requires_grad_(False).eval()
 def export_aux_state(self):
  return dict(model=copy.deepcopy(self.aux_head.state_dict()),optimizer=copy.deepcopy(self.aux_optimizer.state_dict()),steps=self.aux_steps,
   grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in self.aux_head.named_parameters()},
   modes={n:m.training for n,m in self.aux_head.named_modules()},requires_grad={n:p.requires_grad for n,p in self.aux_head.named_parameters()})
 def import_aux_state(self,x):
  self.aux_head.load_state_dict(x['model'],strict=True);self.aux_optimizer.load_state_dict(x['optimizer']);self.aux_steps=x['steps']
  for n,p in self.aux_head.named_parameters():p.grad=None if x['grads'][n] is None else x['grads'][n].to(p.device);p.requires_grad_(x['requires_grad'][n])
  for n,m in self.aux_head.named_modules():m.training=x['modes'][n]
 def teach_aux(self,x,main_step):
  x=self.map_tensor_to_device(x)
  valid=x['effect_valid'].bool() & ~x['effect_holdout'].bool()
  if not valid.any():return dict(updated=False,count=0,nll=None,head_steps=self.aux_steps)
  # The auxiliary head learns its target even during whole-stage migration fallback.
  # Its teacher loss cannot update either the temporal backbone or deployment GMM.
  saved=rng();modes={n:m.training for n,m in self.policy.module.named_modules()}
  try:
   torch.manual_seed(940000000+SEED*1000000+main_step);self.policy.module.eval()
   with torch.no_grad():z=self.policy.module.temporal_encode(feature(x['feature'][valid]))[:,-1].detach()
   self.aux_head.requires_grad_(True).eval();self.aux_optimizer.zero_grad(set_to_none=True)
   loss=-self.aux_head(z).log_prob(x['next_action'][valid]).mean();assert torch.isfinite(loss);loss.backward()
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.aux_head.parameters())
   torch.nn.utils.clip_grad_norm_(self.aux_head.parameters(),self.cfg.train.grad_clip or float('inf'))
   self.aux_optimizer.step();self.aux_steps+=1
   result=dict(updated=True,count=int(valid.sum()),nll=float(loss.detach()),head_steps=self.aux_steps)
   self.aux_optimizer.zero_grad(set_to_none=True)
   return result
  finally:
   self.aux_head.requires_grad_(False).eval()
   for n,m in self.policy.module.named_modules():m.training=modes[n]
   restore_rng(saved)
