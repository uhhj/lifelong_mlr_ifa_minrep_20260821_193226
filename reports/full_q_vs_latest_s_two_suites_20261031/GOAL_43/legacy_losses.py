"""Frozen CCBP-v1 and two execution-matched controls; no trainable additions."""
import torch
import torch.nn.functional as F
GROUPS=((0,3),(3,6),(6,7))
def cost(mu,sd,y,scale):
 return sum(((mu[:,:,a:b]-y[:,None,a:b]).square()+sd[:,:,a:b].square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate(GROUPS))
def w2(mu,sd,tm,ts,scale):
 return sum(((mu[:,:,a:b]-tm[:,None,a:b]).square()+(sd[:,:,a:b]-ts[:,None,a:b]).square()).mean(-1)/scale[g].square() for g,(a,b) in enumerate(GROUPS))
def route(logits,good):
 good=good.detach();allgood=good.all(-1);assert good.any(-1).all()
 # Finite sentinels avoid inf*0 and retain an exact zero for all-good rows.
 hi=logits.masked_fill(~good,-1e9).max(-1).values
 bad=logits.masked_fill(good,-1e9).max(-1).values
 return torch.where(allgood,logits.sum(-1)*0,F.relu(.1+bad-hi))
def aux(arm,q,y,teacher,scale):
 mu,sd,logits=q;tm,ts,tl=teacher;ix=torch.arange(len(y),device=y.device)
 winner=logits.detach().argmax(-1);tk=tl.argmax(-1)
 cc=cost(mu,sd,y,scale);selected=cc[ix,winner]
 budget=cost(tm,ts,y,scale)[ix,tk].detach()+.05
 if arm=='A':return mu.sum()*0,dict(count=len(y))
 if arm=='B':
  d=w2(mu,sd,tm[ix,tk],ts[ix,tk],scale);value=d[ix,winner];good=F.one_hot(d.detach().argmin(-1),5).bool();empty=torch.zeros(len(y),device=y.device,dtype=torch.bool)
 elif arm=='C':
  value=selected;good=F.one_hot(cc.detach().argmin(-1),5).bool();empty=torch.zeros(len(y),device=y.device,dtype=torch.bool)
 else:
  assert arm=='D';good=cc.detach()<=budget[:,None];empty=~good.any(-1)
  value=F.relu(selected-budget)+empty.to(selected.dtype)*F.relu(cc.min(-1).values-budget)
  good=good| (empty[:,None]&F.one_hot(cc.detach().argmin(-1),5).bool())
 rr=route(logits,good);loss=(value+.1*rr).mean()
 return loss,dict(count=len(y),expert_selected_sum=float(selected.detach().sum()),teacher_budget_sum=float(budget.sum()),excess_sum=float(F.relu(selected-budget).detach().sum()),route_sum=float(rr.detach().sum()),empty=int(empty.sum()),all_good=int(good.all(-1).sum()))
