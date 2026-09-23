import contextlib,copy,gc,io,time
import numpy as np,torch
from gradient_relation_experiment import kernel as g
from functional_geometry_diagnostic_v1.geometry import gmm
from .common import *

def ps(algo):return tuple(p for p in algo.policy.parameters() if p.requires_grad)
def flat(items):return torch.cat([x.reshape(-1) for x in items])
def state_equal(a,b):return g.compare_states(a,b)
def proposal(algo,data,base):
 """Transactional native operation: candidate is never committed to source state."""
 g.restore(algo,base);algo.optimizer.zero_grad();data[0]=algo.map_tensor_to_device(data[0])
 loss,_=g.module(algo).compute_loss(data[0],None,None)
 (algo.loss_scale*loss).backward()
 pre=float(torch.nn.utils.clip_grad_norm_(algo.policy.parameters(),algo.cfg.train.grad_clip)) if algo.cfg.train.grad_clip is not None else None
 grads=flat([torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in ps(algo)])
 algo.optimizer_step_and_maybe_schedule();candidate=g.capture(algo)
 d=flat([p.detach().double()-base['model'][n].to(p.device).double() for n,p in algo.policy.named_parameters() if p.requires_grad])
 result=dict(candidate=candidate,d0=d,gradient=grads,loss=float(loss.detach()),preclip_norm=pre)
 g.restore(algo,base);assert all(state_equal(base,g.capture(algo)).values())
 return result

def apply(algo,base,d,alpha):
 offset=0;actual=[]
 with torch.no_grad():
  for n,p in algo.policy.named_parameters():
   if p.requires_grad:
    old=base['model'][n].to(p.device).double();part=d[offset:offset+p.numel()].reshape_as(p)
    p.copy_(old+alpha*part);actual.append((p.detach().double()-old).reshape(-1));offset+=p.numel()
 assert offset==d.numel()
 return torch.cat(actual)

def single_loss(algo,data):
 dist=gmm(algo,data);target=data['actions'].squeeze(1)[:,-1,:]
 return -dist.log_prob(target).mean()*g.module(algo).policy_head.loss_coef

def probe_data(algo,ds,rows):return [algo.map_tensor_to_device(g.batch(ds,[r['record_id']])[0]) for r in rows]

def parity(algo,ds,start):
 base=g.capture(algo);p=proposal(algo,g.batch(ds,start['record_indices']),base)
 g.native(algo,g.batch(ds,start['record_indices']),start['task_id'],start['epoch']);native=g.capture(algo)
 checks=state_equal(p['candidate'],native)
 checks['optimizer_plus_one']=native['optimizer_step']==base['optimizer_step']+1
 checks['scheduler_plus_one']=native['scheduler_step']==base['scheduler_step']+1
 g.restore(algo,base);checks['transaction_restores_source']=all(state_equal(base,g.capture(algo)).values())
 apply(algo,base,p['d0'],1.)
 errors=[];bitwise=True
 for name,parameter in algo.policy.named_parameters():
  if parameter.requires_grad:
   target=native['model'][name].to(parameter.device)
   bitwise=bitwise and torch.equal(parameter,target)
   error=(parameter.detach()-target).abs()
   bound=2*torch.finfo(parameter.dtype).eps*torch.maximum(target.abs(),parameter.detach().abs())+torch.finfo(parameter.dtype).tiny
   errors.append(bool((error<=bound).all()))
 checks['theta_plus_d0_reconstructs_native_within_2ulp']=all(errors)
 g.restore(algo,base)
 assert all(checks.values()),checks
 return dict(start_id=start['start_id'],checks=checks,status='PASS',d0_norm=float(p['d0'].double().norm()),parameter_count=p['d0'].numel(),theta_plus_d0_bitwise=bitwise)

def measure(algo,ds,start,rows,second_order=False):
 begin=time.monotonic();base=g.capture(algo);proposal_data=proposal(algo,g.batch(ds,start['record_indices']),base)
 d0=proposal_data['d0'];grad=proposal_data['gradient'];assert torch.isfinite(d0).all() and torch.isfinite(grad).all();dn=d0.double().norm();gn=grad.double().norm()
 raw=-grad.double()*(dn/gn) if gn>0 else torch.zeros_like(grad)
 assert gn>0 or dn==0,'Zero gradient with nonzero history displacement: raw equal-norm undefined'
 cosine=float(torch.dot(d0.double(),raw.double())/(dn*raw.double().norm())) if dn>0 else None
 data=probe_data(algo,ds,rows);algo.policy.eval();parameters=ps(algo);original_losses=[];dots=[];hnorms=[];repeat=[]
 for dat in data:
  loss=single_loss(algo,dat);original_losses.append(float(loss.detach()))
  hi=torch.autograd.grad(loss,parameters,allow_unused=True)
  h=flat([torch.zeros_like(p) if h is None else h for h,p in zip(hi,parameters)]).detach().double()
  dots.append([float(torch.dot(h,d0.double())),float(torch.dot(h,raw.double()))]);hnorms.append(float(h.norm()))
  with torch.no_grad():repeat.append(float(single_loss(algo,dat)))
  del hi,h,loss
 assert all(np.isfinite(original_losses)) and all(np.isfinite(repeat))
 tolerance=[max(2*abs(a-b),32*np.finfo(np.float32).eps*max(1,abs(a))) for a,b in zip(original_losses,repeat)]
 output=[];dmetadata=[]
 for arm,d in [('adamw',d0),('raw',raw)]:
  for alpha in ALPHAS:
   actual=apply(algo,base,d,alpha);err=float((actual.double()-alpha*d.double()).norm())
   algo.policy.eval()
   with torch.no_grad():values=[float(single_loss(algo,dat)) for dat in data]
   assert all(np.isfinite(values)),(start['start_id'],arm,alpha)
   # Main prediction stays alpha*h.d; rounding diagnostics never replace that target.
   for j,(r,y) in enumerate(zip(rows,values)):
    dot=dots[j][0 if arm=='adamw' else 1];out=dict(start_id=start['start_id'],seed=start['seed'],transition=start['transition'],stage=start['stage'],epoch=start['epoch'],global_step=start['global_step'],task_id_old=r['task_id'],trajectory_id=r['trajectory_id'],window_index=r['window_index'],record_id=r['record_id'],arm=arm,alpha=alpha,h_dot_d0=dots[j][0],h_dot_direction=dot,predicted_delta_loss=alpha*dot,measured_delta_loss=y-original_losses[j],base_loss=original_losses[j],candidate_loss=y,h_norm=hnorms[j],d0_norm=float(dn),direction_norm=float(d.double().norm()),actual_written_norm=float(actual.double().norm()),write_rounding_norm=err,rounding_first_order_bound=hnorms[j]*err,tau_zero=tolerance[j],predicted_deadband=tolerance[j],cosine_adamw_raw=cosine)
    output.append(out)
   dmetadata.append(dict(arm=arm,alpha=alpha,write_rounding_norm=err))
 g.restore(algo,base);restored=state_equal(base,g.capture(algo));assert all(restored.values()),restored
 elapsed=time.monotonic()-begin
 result=dict(start_id=start['start_id'],seconds=elapsed,mixed_bc=proposal_data['loss'],preclip_norm=proposal_data['preclip_norm'],postclip_norm=float(gn),optimizer_before=base['optimizer_step'],optimizer_candidate=proposal_data['candidate']['optimizer_step'],scheduler_before=base['scheduler_step'],scheduler_candidate=proposal_data['candidate']['scheduler_step'],lr=base['optimizer']['param_groups'][0]['lr'],d0_norm=float(dn),raw_norm=float(raw.double().norm()),cosine_adamw_raw=cosine,restored=restored,noise=dict(max_repeat_abs_difference=max(abs(a-b) for a,b in zip(original_losses,repeat)),strictly_repeatable=original_losses==repeat,tau_min=min(tolerance),tau_max=max(tolerance)),formal_training_resumed=False)
 return output,result

def second_order(algo,ds,start,rows):
 base=g.capture(algo);proposed=proposal(algo,g.batch(ds,start['record_indices']),base);d=proposed['d0'];algo.policy.eval();parameters=ps(algo);out=[]
 for r,dat in zip(rows,probe_data(algo,ds,rows)):
  loss=single_loss(algo,dat);base_loss=float(loss.detach());hi=torch.autograd.grad(loss,parameters,create_graph=True,allow_unused=True)
  directional=loss*0;offset=0
  for h,p in zip(hi,parameters):
   part=d[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
   if h is not None:directional=directional+(h*part).sum()
  hv=torch.autograd.grad(directional,parameters,allow_unused=True);second=0.;offset=0
  for h,p in zip(hv,parameters):
   part=d[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
   if h is not None:second+=float((h.double()*part.double()).sum())
  first=float(directional.detach());del loss,hi,hv,directional
  apply(algo,base,d,1.)
  with torch.no_grad():actual=float(single_loss(algo,dat))-base_loss
  apply(algo,base,d,0.)
  out.append(dict(start_id=start['start_id'],seed=start['seed'],transition=start['transition'],stage=start['stage'],task_id_old=r['task_id'],record_id=r['record_id'],first_order=first,hessian_quadratic=second,second_order=first+.5*second,measured_delta_loss=actual))
 g.restore(algo,base);assert all(state_equal(base,g.capture(algo)).values());return out
