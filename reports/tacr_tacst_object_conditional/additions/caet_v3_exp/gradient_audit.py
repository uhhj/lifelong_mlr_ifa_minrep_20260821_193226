import copy,torch
from torch.nn import functional as F
from coreset_exp.model import angular_mean
from .model import hidden,ResidualHead,DeployResidual
def audit_gradients(policy,student,x):
 saved=copy.deepcopy(student.state_dict());modes={n:m.training for n,m in policy.named_modules()};policy.eval()
 try:
  h=hidden(policy,x['feature']);hp,c,g,d=student.deploy(h)
  assert torch.equal(h,hp) and float(d.abs().max())==0 and torch.allclose(g,torch.full_like(g,.1),atol=1e-7)
  assert all(torch.equal(a,b) for a,b in zip(policy.policy_head.forward_fn(h),policy.policy_head.forward_fn(hp)))
  bc=-policy.policy_head(hp).log_prob(x['next_action']).mean()
  ug=torch.autograd.grad(bc,list(student.deploy.U.parameters()),retain_graph=True)
  assert sum(float(z.double().square().sum()) for z in ug)>0
  rg=torch.autograd.grad(bc,list(student.deploy.R.parameters()),retain_graph=True,allow_unused=True)
  assert all(z is None or float(z.abs().max())==0 for z in rg)
  with torch.no_grad():student.deploy.U.weight.normal_(0,.01)
  h=hidden(policy,x['feature']);hp,c,g,d=student.deploy(h)
  groups={'Temporal':list(policy.temporal_transformer.parameters()),'GMM':list(policy.policy_head.parameters()),'R':list(student.deploy.R.parameters()),'U':list(student.deploy.U.parameters()),'gate':list(student.deploy.gate.parameters()),'E':list(student.E.parameters()),'D':list(student.D.parameters())}
  ps=[p for q in groups.values() for p in q]
  own=torch.randn_like(h[0]);old=torch.randn_like(h[0])
  losses={'BC':-policy.policy_head(hp).log_prob(x['next_action']).mean(),'IFA':angular_mean(h,own,[old],.3),'effect':F.smooth_l1_loss(student.D(student.condition(c,x['next_action'])),torch.randn(len(c),7,device=c.device))}
  keys=F.normalize(torch.randn(5,128,device=c.device),dim=-1).detach();z=F.normalize(student.condition(c,x['next_action']),dim=-1);scores=z@keys.T/.1
  losses['ST']=(torch.logsumexp(scores,1)-scores[:,0]).mean()
  expected={'BC':{'Temporal','GMM','R','U','gate'},'IFA':{'Temporal'},'effect':{'R','E','D'},'ST':{'R','E'}};proof={}
  for name,loss in losses.items():
   grads=torch.autograd.grad(loss,ps,allow_unused=True,retain_graph=True);offset=0;norms={}
   for group,q in groups.items():
    gg=grads[offset:offset+len(q)];offset+=len(q);norms[group]=sum(float(v.double().square().sum()) for v in gg if v is not None)
   assert {k for k,v in norms.items() if v>0}==expected[name],(name,norms);proof[name]=norms
  # The same residual wrapper that formal CompletePolicy uses must match training fusion.
  copybranch=DeployResidual().to(h.device);copybranch.load_state_dict(student.deploy.state_dict(),strict=True)
  wrapped=ResidualHead(policy.policy_head,copybranch)
  assert all(torch.equal(a,b) for a,b in zip(wrapped.forward_fn(h),policy.policy_head.forward_fn(hp)))
  assert len(copybranch.state_dict())==14
  return dict(gradient_norm2=proof,initial_actual_GMM_parameters_exact=True,initial_U_BC_gradient_nonzero=True,initial_R_BC_gradient_zero_expected=True,deploy_adapter_same_fusion_exact=True,strict_deploy_keys=14,probe='isolated actual core/GMM with constructed ST keys; actual qualified graph separately audited')
 finally:
  student.load_state_dict(saved,strict=True)
  for n,m in policy.named_modules():m.training=modes[n]
