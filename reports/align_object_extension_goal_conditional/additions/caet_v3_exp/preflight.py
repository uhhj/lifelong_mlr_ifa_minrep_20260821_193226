"""Per-entry actual isolated GPU optimization; no environment."""
import copy,json,math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,SEED,write,save,verify
from coreset_exp.model import cfg
from coreset_exp.train import PaperMLR
from libero.lifelong.algos.mlr import build_dataset_for_task
from .train import build,capture,restore
from .dataset import EffectDataset
from .model import hidden,masked_bc,params,selected,kl
from .variant import VARIANT
from .eftr_optimizer import choose
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b
def main():
 verify();assert not (O/'GPU_PREFLIGHT.json').exists();dist.init_process_group('nccl');torch.cuda.set_device(0);calls=[];proof={}
 try:
  c=cfg('ra_coreset');c.run_root=str(O/'preflight/run');c.buffer_dir=str(O/'preflight/run/buffer');c.experiment_dir=str(O/'preflight/run/models')
  a,_=build(c);ds=EffectDataset(build_dataset_for_task(c.feature_dir,6,c.buffer_dir,SEED,start=6),6);loader=DataLoader(ds,batch_size=10,num_workers=0)
  a.start_task(-1,loader);a.refs={6:ds.native[0][0]['feature'][0,0,0].cuda()};a.configure_effect(6,ds)
  def hook(opt,name):
   original=opt.step
   def step(*args,**kwargs):
    v=original(*args,**kwargs);calls.append(dict(index=len(calls)+1,kind=name));write(O/'preflight/optimizer_calls.json',calls);return v
   opt.step=step
  hook(a.optimizer,'main')
  if a.auxopt is not None:hook(a.auxopt,'auxiliary')
  batch=next(iter(loader));info={'selected_pairs':[]};initial=capture(a)
  PaperMLR.observe(a,copy.deepcopy(batch),6,0,info);native=capture(a)
  restore(a,initial);a.disable_all=True;a.observe(copy.deepcopy(batch),6,0,info);assert equal(native,capture(a));a.disable_all=False
  proof['disabled_full_state_exact_RA']=True
  x=a.map_tensor_to_device(copy.deepcopy(batch[0]));p=a.policy.module;p.eval();h=hidden(p,x['feature']);loss,v=masked_bc(p,h,x,True)
  expected=-p.policy_head(h).log_prob(x['next_action'])[v].mean()*p.policy_head.loss_coef
  assert torch.equal(loss,expected);proof['masked_valid_nll_exact']=True
  xx=dict(x,causal_bc_valid=torch.zeros_like(v));zero,_=masked_bc(p,h,xx,True);assert float(zero)==0 and zero.requires_grad
  proof['empty_valid_mask_differentiable_zero']=True
  if VARIANT=='TAC':
   em=x['effect_valid'].bool();scale=h.new_tensor([.01]*3+[math.pi/18]*3+[.01]);effect=torch.nn.functional.smooth_l1_loss(a.aux.predict(h[em],x['next_action'][em]),x['effect_target'][em]/scale)
   pp=list(p.temporal_transformer.parameters())+list(p.policy_head.parameters());g=torch.autograd.grad(effect,pp,allow_unused=True,retain_graph=True);nt=len(list(p.temporal_transformer.parameters()))
   norm=sum(float(z.double().square().sum()) for z in g[:nt] if z is not None);assert norm>0 and all(z is None for z in g[nt:])
   emb=a.aux.embedding(h[:3]);con=torch.logsumexp((emb[0]@emb[1:].T)/.1,0)-(emb[0]@emb[1])/.1
   gg=torch.autograd.grad(con,pp,allow_unused=True);cn=sum(float(z.double().square().sum()) for z in gg[:nt] if z is not None);assert cn>0 and all(z is None for z in gg[nt:])
   proof.update(effect_Temporal_gradient_norm2=norm,contrast_Temporal_gradient_norm2=cn,auxiliary_direct_GMM_gradient_absent=True,contrast_probe='controlled three actual feature rows; scientific pairs separately zero-training audited')
  if VARIANT=='EFTR':
   ids=list(range(10));vv=params(p,ds.features(ids));teacher={}
   for k,i in enumerate(ids):
    r=ds.rows[i];teacher['t6__'+r['sample']]=dict(sample_sha256=r['sample_sha256'],means=vv[0][k].detach(),std=vv[1][k].detach(),logits=vv[2][k].detach(),mode=int(vv[2][k].argmax()))
   a.old=ids;a.teacher=teacher
   ref=selected(vv);bad=(ref[0]+.3,ref[1]*2,ref[2]);assert float(kl(ref,bad).mean())>float(kl(ref,ref).mean())
   assert choose([dict(C=1.,R=.4),dict(C=1.01,R=.03),dict(C=1.04,R=0.)])[0]==1
   assert choose([dict(C=1.,R=.4),dict(C=1.03,R=.03),dict(C=1.04,R=0.)])==(0,True)
   proof.update(harmful_exec_distribution_detected=True,candidate_plasticity_and_budget_rules=True,teacher_probe='isolated current rows treated as historical only within preflight')
  restore(a,native);a.policy.train();a.observe(copy.deepcopy(batch),6,0,info);recovery=capture(a)
  save(O/'preflight/recovery.pth',recovery)
  a.observe(copy.deepcopy(batch),6,0,info);after=capture(a);after_stat=copy.deepcopy(a.caet_epoch_statistics[-1])
  restore(a,torch.load(O/'preflight/recovery.pth',map_location='cpu'));a.observe(copy.deepcopy(batch),6,0,info)
  assert equal(after,capture(a));proof['active_serialized_full_state_restore_exact']=True
  if VARIANT=='EFTR':
   assert after_stat['optimizer_trials']==3 and a.caet_epoch_statistics[-1]['scores']==after_stat['scores']
   proof['actual_three_candidate_replay_exact']=True
  assert len(a.policy.module.state_dict())==100
  restore(a,initial)
 finally:dist.destroy_process_group()
 verify();write(O/'GPU_PREFLIGHT.json',dict(status='PASS',variant=VARIANT,optimizer_calls=len(calls),main_optimizer_calls=sum(x['kind']=='main' for x in calls),auxiliary_optimizer_calls=sum(x['kind']=='auxiliary' for x in calls),formal_steps=0,environment_steps=0,proof=proof,gpu=torch.cuda.get_device_name()))
if __name__=='__main__':main()
