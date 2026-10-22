"""Actual isolated GPU updates and full recovery; zero environments."""
import copy,json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,SEED,write,save,verify,rng,restore_rng
from coreset_exp.model import cfg
from coreset_exp.train import PaperMLR
from libero.lifelong.algos.mlr import build_dataset_for_task
from .train import build,capture,restore
from .dataset import EffectDataset
from .model import hidden,masked_bc
from .variant import VARIANT
from .gradient_audit import audit_gradients

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def main():
 verify();assert not (O/'GPU_PREFLIGHT.json').exists();dist.init_process_group('nccl');torch.cuda.set_device(0);calls=[]
 try:
  c=cfg('ra_coreset');c.run_root=str(O/'preflight/run');c.buffer_dir=str(O/'preflight/run/buffer');c.experiment_dir=str(O/'preflight/run/models')
  a,_=build(c);ds=EffectDataset(build_dataset_for_task(c.feature_dir,6,c.buffer_dir,SEED,start=6),6);loader=DataLoader(ds,batch_size=10,num_workers=0)
  a.start_task(-1,loader);a.refs={6:ds.native[0][0]['feature'][0,0,0].cuda()};a.configure_effect(6,ds)
  def hook(opt,name):
   original=opt.step
   def step(*args,**kwargs):
    v=original(*args,**kwargs);calls.append(dict(index=len(calls)+1,kind=name));write(O/'preflight/optimizer_calls.json',calls);return v
   opt.step=step
  hook(a.optimizer,'main');hook(a.repopt,'representation')
  batch=next(iter(loader));info={'selected_pairs':[]};initial=capture(a)
  PaperMLR.observe(a,copy.deepcopy(batch),6,0,info);native=capture(a)
  restore(a,initial);a.disable_all=True;a.observe(copy.deepcopy(batch),6,0,info);assert equal(native,capture(a));a.disable_all=False
  proof=dict(disabled_full_state_exact_RA=True)
  restore(a,initial);x=a.map_tensor_to_device(copy.deepcopy(batch[0]));saved=rng()
  proof.update(audit_gradients(a.policy.module,a.student,x));restore_rng(saved)
  a.policy.eval();h=hidden(a.policy.module,x['feature']);bc,v=masked_bc(a.policy.module,h,x)
  expected=-a.policy.module.policy_head(h).log_prob(x['next_action'])[v].mean()*a.policy.module.policy_head.loss_coef
  assert torch.equal(bc,expected);xx=dict(x,causal_bc_valid=torch.zeros_like(v));zero,_=masked_bc(a.policy.module,h,xx);assert float(zero)==0 and zero.requires_grad
  proof.update(masked_valid_nll_exact=True,empty_valid_mask_differentiable_zero=True)
  restore(a,native)
  if VARIANT=='TACRST':
   # Isolated synthetic ST activation tests cursor/immutable keys; actual graph and stage teacher audited separately.
   a.task=7;ds.positive={0:[1]};ds.negative={0:[2,3,4,5]}
   a.keys={i:torch.nn.functional.normalize(torch.randn(128),dim=0).detach() for i in range(1,6)}
   a.teacher_hash={'preflight':'constructed immutable keys only'}
   frozen_keys=copy.deepcopy(a.keys)
  a.policy.train();a.observe(copy.deepcopy(batch),6,0,info);recovery=capture(a);save(O/'preflight/recovery.pth',recovery)
  a.observe(copy.deepcopy(batch),6,0,info);after=capture(a);stat=copy.deepcopy(a.caet_epoch_statistics[-1])
  restore(a,torch.load(O/'preflight/recovery.pth',map_location='cpu'));a.observe(copy.deepcopy(batch),6,0,info)
  assert equal(after,capture(a));assert stat=={**a.caet_epoch_statistics[-1],'seconds':stat['seconds']}
  proof['active_serialized_full_state_restore_exact']=True;proof['student_and_both_optimizer_schedulers_grads_modes_RNG_restored']=True
  assert len(a.policy.module.state_dict())==100 and len(a.student.deploy.state_dict())==14
  if VARIANT=='TACRST':
   assert equal(frozen_keys,a.keys) and a.cursor>0 and a.caet_epoch_statistics[-1]['cross_anchors']>0
   proof['constructed_active_ST_keys_immutable_and_cursor_restored']=True
   a.teacher_hash={};a.task=6
  restore(a,initial)
 finally:dist.destroy_process_group()
 verify();write(O/'GPU_PREFLIGHT.json',dict(status='PASS',variant=VARIANT,optimizer_calls=len(calls),main_optimizer_calls=sum(x['kind']=='main' for x in calls),representation_optimizer_calls=sum(x['kind']=='representation' for x in calls),formal_steps=0,environment_steps=0,proof=proof,gpu=torch.cuda.get_device_name()))
if __name__=='__main__':main()
