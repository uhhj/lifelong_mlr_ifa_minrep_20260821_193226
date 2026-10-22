"""Actual PyTorch AdamW candidates, committed state includes moments."""
import copy,math
import torch
def snapshot(a):
 return copy.deepcopy(dict(model=a.policy.state_dict(),optimizer=a.optimizer.state_dict(),grads={n:p.grad for n,p in a.policy.named_parameters()}))
def restore(a,s):
 a.policy.load_state_dict(s['model'],strict=True);a.optimizer.load_state_dict(s['optimizer'])
 for n,p in a.policy.named_parameters():p.grad=None if s['grads'][n] is None else s['grads'][n].clone().to(p.device)
def apply(a,native,aux,scale):
 a.optimizer.zero_grad()
 for p,n,g in zip([p for p in a.policy.parameters() if p.requires_grad],native,aux):
  if n is None and (scale==0 or g is None):p.grad=None;continue
  p.grad=torch.zeros_like(p) if n is None else n.detach().clone()
  if scale and g is not None:p.grad.add_(g.detach(),alpha=scale)
 if a.cfg.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip)
 a.optimizer.step()
def choose(scores):
 assert all(all(x is None or math.isfinite(x) for x in s.values()) for s in scores)
 c0=scores[0]['C'];valid=[i for i,s in enumerate(scores) if c0 is None or s['C']<=c0+.02]
 assert 0 in valid
 budget=[i for i in valid if scores[i]['R'] is None or scores[i]['R']<=.10]
 eligible=budget or valid
 return min(eligible,key=lambda i:(scores[i]['R'] or 0.,scores[i]['C'] or 0.,i)),not bool(budget)
def candidates(a,native,aux,score):
 start=snapshot(a);states=[];scores=[]
 for scale in (0.,.5,1.):
  restore(a,start);apply(a,native,aux,scale);scores.append(score());states.append(snapshot(a))
 k,unmet=choose(scores);restore(a,states[k]);a.scheduler.step();a.step+=1
 return dict(selected=k,scale=(0.,.5,1.)[k],scores=scores,budget_unmet=unmet,optimizer_trials=3,committed_updates=1)
