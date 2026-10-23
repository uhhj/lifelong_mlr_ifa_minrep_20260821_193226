import torch
from .model import selected,masked_mean
def distance(q,reference):
 mu,sd,_=q;rm,rs,_=reference;d=(mu-rm).square()+(sd-rs).square();return .5*d[...,:6].mean(-1)+.5*d[...,6]
def retain_loss(policy,z,b,cache,current_task):
 ix=b['index'].detach().cpu();refs=[tuple(v[ix].to(z.device) for v in cache[k]) for k in ('qgt','qpred')]
 qgt=policy.decoder(b['effect'],b['feature']);qpred=policy.decode_z(z,b['feature']);mask=b['action_mask']&(b['task']<current_task)[:,None]
 loss=z.sum()*0;route=z.sum()*0;switch=0
 for q,ref in zip((qgt,qpred),refs):
  hard=selected(q);target=selected(ref);loss=loss+.5*masked_mean(distance(hard,target),mask)
  d=distance(q,(target[0].unsqueeze(-2),target[1].unsqueeze(-2),None)).detach();route=route+.5*masked_mean((torch.softmax(q[2],-1)*d).sum(-1),mask)
  switch+=int(((hard[2]!=target[2])&mask).sum())
 return loss,route,int(mask.sum()),switch
