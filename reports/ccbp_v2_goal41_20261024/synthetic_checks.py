"""Numerical/gradient admission; synthetic CPU tensors, zero optimizer/environment calls."""
import torch
from losses import aux,ranking,cost
from prototype_loss import auxiliary_v2,estimate_teacher_cost_median
from legacy_losses import aux as old_aux
def run():
 torch.manual_seed(20261024);scale=torch.tensor([.46725568,.1,1.]);y=torch.randn(3,7)
 mu=torch.randn(3,5,7,requires_grad=True);sd=torch.full_like(mu,.2,requires_grad=True);z=torch.randn(3,5,requires_grad=True)
 teacher=(torch.randn_like(mu,requires_grad=True),torch.ones_like(sd,requires_grad=True),torch.randn_like(z,requires_grad=True));w=torch.tensor([.1,.5,.9])
 for arm in 'BC':
  new=aux(arm,(mu,sd,z),y,teacher,scale,w)[0];old=old_aux(arm,(mu,sd,z),y,teacher,scale)[0];assert torch.equal(new,old)
  gn=torch.autograd.grad(new,(mu,sd,z),retain_graph=True);go=torch.autograd.grad(old,(mu,sd,z),retain_graph=True);assert all(torch.equal(a,b) for a,b in zip(gn,go))
 for arm in 'NV':
  loss=aux(arm,(mu,sd,z),y,teacher,scale,w)[0];g=torch.autograd.grad(loss,(mu,sd,z,*teacher),allow_unused=True,retain_graph=True)
  assert all(t is not None and torch.isfinite(t).all() for t in g[:3]) and all(t is None for t in g[3:])
 tau=estimate_teacher_cost_median(teacher,y,scale);ct=cost(teacher[0],teacher[1],y,scale).gather(1,teacher[2].argmax(-1)[:,None]).squeeze(1).detach();ww=tau/(tau+ct)
 for arm,mode in [('N','none'),('V','quality')]:
  got=aux(arm,(mu,sd,z),y,teacher,scale,ww)[0];expected=auxiliary_v2((mu,sd,z),y,teacher,scale,teacher_cost_median=tau,anchor_mode=mode)[0];assert torch.equal(got,expected)
  gg=torch.autograd.grad(got,(mu,sd,z),retain_graph=True);ge=torch.autograd.grad(expected,(mu,sd,z),retain_graph=True);assert all(torch.equal(a,b) for a,b in zip(gg,ge))
 # Low expert error still supplies execution regression gradients.
 a=torch.full((1,5,7),.001,requires_grad=True);s=torch.full_like(a,.001,requires_grad=True);l=torch.tensor([[2.,0.,-1.,-2.,-3.]],requires_grad=True)
 t=(torch.zeros_like(a),torch.full_like(s,.001),l.detach());zero=torch.zeros(1,7)
 ex=aux('V',(a,s,l),zero,t,torch.ones(3),torch.ones(1)*.5)[2]
 assert torch.autograd.grad(ex,a)[0].abs().sum()>0
 # Larger fixed score gap => strictly larger routing correction at identical logits.
 zg=torch.zeros(1,5,requires_grad=True);small=torch.tensor([[0.,1.,0.,0.,0.]]);big=small*4
 gs=torch.autograd.grad(ranking(zg,small).sum(),zg,retain_graph=True)[0];gb=torch.autograd.grad(ranking(zg,big).sum(),zg)[0];assert torch.allclose(gb,4*gs)
 # Cost ties choose the highest logit, matching the submitted prototype.
 score=torch.tensor([[0.,0.,2.,3.,4.]],requires_grad=True);zz=torch.tensor([[.7,.8,1.,-.1,.3]],requires_grad=True);perm=torch.tensor([4,1,3,0,2])
 r=ranking(zz,score).sum();g=torch.autograd.grad(r,(zz,score),allow_unused=True);assert g[1] is None
 zp=zz.detach()[:,perm].requires_grad_();rp=ranking(zp,score.detach()[:,perm]).sum();gp=torch.autograd.grad(rp,zp)[0]
 assert torch.allclose(r,rp) and torch.allclose(g[0][:,perm],gp)
 q=(mu,sd,z);permuted=tuple(t[:,perm] for t in q);tp=tuple(t[:,perm] for t in teacher)
 for arm in 'NV':assert torch.allclose(aux(arm,q,y,teacher,scale,w)[0],aux(arm,permuted,y,tp,scale,w)[0],atol=1e-5,rtol=1e-6)
 assert ranking(torch.zeros(1,5),torch.ones(1,5)).item()==0
 return dict(status='PASS',uploaded_prototype_loss_gradient_exact=True,control_BC_loss_gradient_exact=True,teacher_no_grad=True,low_cost_execution_gradient=True,cost_gap_routing_gradient=True,minimum_tie_and_permutation=True,optimizer_updates=0,environments=0)
