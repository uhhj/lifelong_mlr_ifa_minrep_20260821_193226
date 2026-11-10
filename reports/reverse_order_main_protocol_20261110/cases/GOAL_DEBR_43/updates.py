from runtime import *
from losses import expert
from legacy_losses import aux as legacy_aux
from support_memory import retention

def update(a,batch,arm,memory,scale,info,stage,measure=False,retain_weight=RETAIN_WEIGHT,legacy=False):
    x=a.map_tensor_to_device(batch[0]);tasks=torch.tensor([int(v) for v in batch[1]],device='cuda');a.optimizer.zero_grad()
    h=hidden(a.policy.module,x['feature']);bc,valid=masked_bc(a.policy.module,h,x,True)
    mask=tasks==stage;olds=[a.refs[j if k==stage else k] for j,k in info['selected_pairs']]
    ifa=angular_mean(h[mask],a.refs[stage],olds,a.cfg.gamma) if mask.any() else h.sum()*0
    eligible=(tasks<stage)&valid;ex=h.sum()*0;rr=ex;rx=ex;rt=ex;extra=ex
    anchor=dict(slots=0,actual=0,empty_slots=0,forward_calls=0,selection=[])
    if eligible.any() or stage>6:
        saved=rng();modes={n:m.training for n,m in a.policy.named_modules()};a.policy.eval()
        try:
            if eligible.any():
                q=params(a.policy.module,x['feature'][eligible]);y=x['next_action'][eligible]
                extra,ex,rr=expert(q,y,scale)
                if legacy:extra,_=legacy_aux('C',q,y,tuple(v.detach() for v in q),scale)
            if stage>6 and retain_weight!=0:
                rx,rt,anchor=retention(a.policy.module,memory,stage,a.step,arm)
        finally:
            for n,m in a.policy.named_modules():m.training=modes[n]
            restore_rng(saved)
    grad=None
    if measure and eligible.any():
        saved=rng();pars=[p for p in a.policy.parameters() if p.requires_grad]
        before=[None if p.grad is None else p.grad.detach().clone() for p in pars];grad={};tic=time.time()
        for name,term in [('bc',bc),('ifa_weighted',.1*ifa),('expert_weighted',.1*ex),('expert_route_weighted',.01*rr),('retain_weighted',retain_weight*rx),('retain_route_weighted',retain_weight*.1*rt)]:
            gs=torch.autograd.grad(a.loss_scale*term,pars,retain_graph=True,allow_unused=True)
            grad[name]=dict(loss=float(term.detach()),l2=float(torch.sqrt(sum((g.double().square().sum() for g in gs if g is not None),torch.zeros((),device='cuda',dtype=torch.float64)))))
        assert equal(saved,rng()) and all((p.grad is None if g is None else p.grad is not None and torch.equal(p.grad,g)) for p,g in zip(pars,before))
        grad.update(seconds=time.time()-tic,extra_autograd_calls=6)
    loss=bc+.1*ifa+.1*extra+retain_weight*(rx+.1*rt)
    (a.loss_scale*loss).backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())
    torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip);a.optimizer.step();a.scheduler.step();a.step+=1
    return dict(loss=float(loss.detach()),bc=float(bc.detach()),ifa=float(ifa.detach()),expert_loss=float(ex.detach()),expert_route=float(rr.detach()),retain_loss=float(rx.detach()),retain_route=float(rt.detach()),valid=int(valid.sum()),invalid=int((~valid).sum()),old_eligible=int(eligible.sum()),anchor_slots=anchor['slots'],anchor_actual=anchor['actual'],anchor_empty=anchor['empty_slots'],anchor_forwards=anchor['forward_calls'],gradient_diagnostic=grad)
