from runtime import *
from losses import aux
from legacy_losses import aux as legacy_aux
def update(a,batch,arm,cache,scale,weights,info,stage,measure=False,legacy=False):
 x=a.map_tensor_to_device(batch[0]);tasks=torch.tensor([int(v) for v in batch[1]],device='cuda');a.optimizer.zero_grad()
 h=hidden(a.policy.module,x['feature']);bc,valid=masked_bc(a.policy.module,h,x,True)
 mask=tasks==stage;olds=[a.refs[j if k==stage else k] for j,k in info['selected_pairs']]
 ifa=angular_mean(h[mask],a.refs[stage],olds,a.cfg.gamma) if mask.any() else h.sum()*0
 eligible=(tasks<stage)&valid;extra=h.sum()*0;ex=extra;rr=extra;st=dict(count=0)
 if eligible.any():
  saved=rng();modes={n:m.training for n,m in a.policy.named_modules()};a.policy.eval()
  try:
   q=params(a.policy.module,x['feature'][eligible]);ids=x['v3_index'][eligible].tolist();teacher=tuple(torch.stack([cache[i][j] for i in ids]).cuda() for j in range(3));w=torch.tensor([weights[i] for i in ids],device='cuda')
   extra,st,ex,rr=aux(arm,q,x['next_action'][eligible],teacher,scale,w)
   if legacy:extra,_=legacy_aux(arm,q,x['next_action'][eligible],teacher,scale)
  finally:
   for n,m in a.policy.named_modules():m.training=modes[n]
   restore_rng(saved)
 grad=None
 if measure and eligible.any():
  saved=rng();pars=[p for p in a.policy.parameters() if p.requires_grad];before_grads=[None if p.grad is None else p.grad.detach().clone() for p in pars];grad={};tic=time.time()
  for name,term in [('bc',bc),('ifa_weighted',.1*ifa),('exec_weighted',.1*ex),('route_weighted',.01*rr)]:
   gs=torch.autograd.grad(a.loss_scale*term,pars,retain_graph=True,allow_unused=True)
   grad[name]=dict(loss=float(term.detach()),l2=float(torch.sqrt(sum((g.double().square().sum() for g in gs if g is not None),torch.zeros((),device='cuda',dtype=torch.float64)))),nonzero_tensors=sum(g is not None and bool(g.abs().sum()>0) for g in gs))
  assert all((p.grad is None if g is None else p.grad is not None and torch.equal(p.grad,g)) for p,g in zip(pars,before_grads)) and equal(saved,rng());grad['seconds']=time.time()-tic;grad['extra_autograd_calls']=4
 loss=bc+.1*ifa+.1*extra;(a.loss_scale*loss).backward()
 assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())
 torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip);a.optimizer.step();a.scheduler.step();a.step+=1
 return dict(loss=float(loss.detach()),bc=float(bc.detach()),ifa=float(ifa.detach()),aux=float(extra.detach()),exec_loss=float(ex.detach()),route_loss=float(rr.detach()),valid=int(valid.sum()),invalid=int((~valid).sum()),old_eligible=int(eligible.sum()),aux_stats=st,gradient_diagnostic=grad)
