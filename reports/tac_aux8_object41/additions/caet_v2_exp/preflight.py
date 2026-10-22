"""Actual isolated GPU updates, gradient routing and complete-state restoration."""
import copy,json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,write,save,verify
from coreset_exp.model import cfg
from coreset_exp.train import PaperMLR,capture as native_capture
from libero.lifelong.algos.mlr import build_dataset_for_task
from .train import build,capture,restore
from .dataset import EffectDataset
from .functional_reference import nll,dual
from .model import EffectModel
from .observe import safe_transfer

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def main():
 verify();assert not (O/'GPU_PREFLIGHT.json').exists();ledger=[]
 def rec(name,kind='main'):
  ledger.append(dict(index=len(ledger)+1,name=name,kind=kind));write(O/'preflight/optimizer_calls.json',ledger)
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  c=cfg('ra_coreset');c.run_root=str(O/'preflight/smoke');c.buffer_dir=str(O/'preflight/smoke/buffer');c.experiment_dir=str(O/'preflight/smoke/models')
  a,_=build(c);native=build_dataset_for_task(c.feature_dir,6,c.buffer_dir,41,start=6);ds=EffectDataset(native,6)
  loader=DataLoader(ds,batch_size=10,num_workers=0);a.start_task(-1,loader);a.refs={6:native[0][0]['feature'][0,0,0].cuda()};a.configure_effect(6,ds)
  batch=next(iter(loader));initial=capture(a)
  one=PaperMLR.observe(a,copy.deepcopy(batch),6,0,{'selected_pairs':[]});rec('native_off_reference');expected=native_capture(a)
  restore(a,initial);two=a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});rec('aux8_fallback_primary');rec('aux8_teacher_fallback','head')
  assert one==two and equal(expected,native_capture(a)), 'Auxiliary-only teacher disturbed native state'
  assert a.aux_steps==1
  save(O/'preflight/recovery.pth',capture(a));a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});rec('uninterrupted');rec('teacher_uninterrupted','head');after=capture(a)
  restore(a,torch.load(O/'preflight/recovery.pth',map_location='cpu'));a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});rec('restored');rec('teacher_restored','head');assert equal(after,capture(a))
  restore(a,initial);policy=a.policy.module;policy.eval();params=[p for p in policy.parameters() if p.requires_grad];names=[n for n,p in policy.named_parameters() if p.requires_grad]
  ids=[i for i,r in enumerate(ds.rows) if r['valid'] and not r['holdout']][:10]
  x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ids]).cuda();actions=torch.stack([ds.rows[i]['next_action'] for i in ids]).cuda();assert x.shape[1:]==(8,4,512)
  eff=EffectModel().cuda().eval().requires_grad_(False);z=policy.temporal_encode(x)[:,-1];means,std,logits=a.aux_head.forward_fn(z);winner=logits.detach().argmax(-1);ix=torch.arange(len(x),device=x.device);act=means[ix,winner]+std[ix,winner]*torch.randn_like(means[ix,winner]);pred,_=eff(x,act)
  eg=torch.autograd.grad(pred.square().mean(),params,allow_unused=True)
  route=dict(temporal_norm2=sum(float(g.double().square().sum()) for n,g in zip(names,eg) if g is not None and n.startswith('temporal_transformer.')),deployment_head_gradients_absent=all(g is None for n,g in zip(names,eg) if n.startswith('policy_head.')),auxiliary_head_frozen_during_transfer=all(not p.requires_grad for p in a.aux_head.parameters()))
  assert route['temporal_norm2']>0 and np.isfinite(route['temporal_norm2']) and route['deployment_head_gradients_absent']
  base=nll(policy,x,actions,a.aux_head).mean();bound=float(base.detach());g=torch.autograd.grad(base,params,allow_unused=True);norm=sum(float(v.double().square().sum()) for v in g if v is not None);assert norm>0
  assert all(v is None for n,v in zip(names,g) if n.startswith('policy_head.'))
  opt=torch.optim.SGD(params,lr=.1/norm);opt.zero_grad()
  for p,v in zip(params,g):p.grad=None if v is None else -v.detach().clone()
  opt.step();rec('constructed_auxiliary_likelihood_harm');harm=nll(policy,x,actions,a.aux_head).mean();hv=float(harm.detach());assert hv>bound+.05,(bound,hv)
  mu=dual({'g':0.},{'g':hv},{'g':bound})['g'];rg=torch.autograd.grad(mu*torch.relu(harm-bound-.05),params,allow_unused=True);rn=sum(float(v.double().square().sum()) for v in rg if v is not None);assert rn>0
  opt=torch.optim.SGD(params,lr=.02*mu/rn);opt.zero_grad()
  for p,v in zip(params,rg):p.grad=None if v is None else v.detach().clone()
  opt.step();rec('auxiliary_protection_response');repaired=float(nll(policy,x,actions,a.aux_head).mean().detach());assert repaired<hv
  plasticity=[]
  for sign in (-1,1):
   restore(a,initial);policy.eval();bx=batch[0]['feature'].cuda();ba=batch[0]['actions'].squeeze(1)[:,-1].cuda();loss=nll(policy,bx,ba).mean();bg=torch.autograd.grad(loss,params,allow_unused=True)
   candidate=[None if v is None else sign*v for v in bg];safe,dot=safe_transfer(candidate,bg);assert (dot<0)==(sign<0)
   norm=sum(float(v.double().square().sum()) for v in bg if v is not None);opt=torch.optim.SGD(params,lr=.01/norm);opt.zero_grad()
   for p,g,v in zip(params,bg,safe):p.grad=None if g is None else g.detach().clone()+(0 if v is None else .05*v.detach())
   opt.step();rec('native_BC_plasticity_'+str(sign));new=float(nll(policy,bx,ba).mean().detach());assert new<float(loss.detach());plasticity.append(dict(dot=dot,before=float(loss.detach()),after=new))
  restore(a,initial);a.effect=eff;a.effect_center=torch.zeros(7,device='cuda');a.effect_scale=torch.ones(7,device='cuda');a.effect_sha256='ISOLATED_PREFLIGHT_ONLY'
  forced=copy.deepcopy(batch);forced[0]['transfer_supported']=forced[0]['effect_valid'] & ~forced[0]['effect_holdout'];active0=capture(a);assert forced[0]['transfer_supported'].any()
  a.observe(copy.deepcopy(forced),6,0,{'selected_pairs':[]});rec('active_effect_route');rec('teacher_active_route','head');active1=capture(a)
  restore(a,active0);a.observe(copy.deepcopy(forced),6,0,{'selected_pairs':[]});rec('active_effect_route_restored');rec('teacher_active_restored','head');assert equal(active1,capture(a))
  assert len(policy.state_dict())==100 and not any('aux' in k for k in policy.state_dict())
  a.effect=None;a.effect_sha256=None;restore(a,initial)
 finally:dist.destroy_process_group()
 verify();assert len(ledger)==15
 write(O/'GPU_PREFLIGHT.json',dict(status='PASS',optimizer_calls=len(ledger),main_optimizer_calls=10,auxiliary_head_optimizer_calls=5,formal_steps=0,environment_steps=0,zero_aux_native_update_exact=True,full_state_epoch_recovery_exact=True,active_aux_recovery_exact=True,auxiliary_gradient_route=route,main_policy_keys=100,auxiliary_not_in_deployment_state_dict=True,retention=dict(bound=bound,harmful=hv,after_response=repaired,mu=mu),plasticity=plasticity,gpu=torch.cuda.get_device_name(),torch=torch.__version__))
if __name__=='__main__':main()
