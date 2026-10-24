"""Fixed B/C controls, N=rho0 and V=rho1; no budget or threshold scan."""
import torch
import torch.nn.functional as F
from legacy_losses import cost,w2,route
def ranking(logits,score):
 s=score.detach();minimum=s.min(-1).values;ties=s==minimum[:,None]
 # Exact uploaded prototype tie semantics; highest-logit minimum.
 zref=logits.masked_fill(~ties,-torch.inf).max(-1).values
 gap=s-minimum[:,None]
 return (gap*F.softplus(logits-zref[:,None])).sum(-1)/(logits.shape[-1]-1)
def aux(arm,q,y,teacher,scale,weight):
 mu,sd,z=q;tm,ts,tz=(t.detach() for t in teacher);ix=torch.arange(len(y),device=y.device)
 k=z.detach().argmax(-1);tk=tz.argmax(-1);c=cost(mu,sd,y,scale)
 distance=w2(mu,sd,tm[ix,tk],ts[ix,tk],scale)
 if arm=='B':
  score=distance;ex=score[ix,k];rr=route(z,F.one_hot(distance.detach().argmin(-1),5).bool())
 elif arm=='C':
  score=c;ex=score[ix,k];rr=route(z,F.one_hot(c.detach().argmin(-1),5).bool())
 else:
  assert arm in 'NV';w=weight.detach() if arm=='V' else torch.zeros_like(weight)
  score=(c+w[:,None]*distance)/(1+w[:,None]);ex=score[ix,k];rr=ranking(z,score)
 loss=(ex+.1*rr).mean()
 stats=dict(count=len(y),expert_selected_sum=float(c[ix,k].detach().sum()),anchor_selected_sum=float(distance[ix,k].detach().sum()),score_selected_sum=float(score[ix,k].detach().sum()),score_min_sum=float(score.detach().min(-1).values.sum()),route_sum=float(rr.detach().sum()),regret_sum=float((score[ix,k]-score.min(-1).values).detach().sum()),weight_sum=float(weight.detach().sum()) if arm=='V' else 0.)
 return loss,stats,ex.mean(),rr.mean()
