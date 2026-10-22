"""Eight isolated policy optimizer calls. Zero environment / formal updates."""
import copy,json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,SEED,write,save,verify
from coreset_exp.model import cfg
from coreset_exp.train import build,PaperMLR,capture,restore
from libero.lifelong.algos.mlr import build_dataset_for_task
from .dataset import EffectDataset
from .observe import CAETv2MLR,safe_transfer
from .functional_reference import nll,dual

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def main():
 verify();assert not (O/'GPU_PREFLIGHT.json').exists();ledger=[]
 def record(name):
  ledger.append(dict(index=len(ledger)+1,name=name));write(O/'preflight/optimizer_calls.json',ledger);assert len(ledger)<=8
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  c=cfg('ra_coreset');c.run_root=str(O/'preflight/smoke');c.buffer_dir=str(O/'preflight/smoke/buffer');c.experiment_dir=str(O/'preflight/smoke/models')
  a,_=build(c);a.__class__=CAETv2MLR
  native=build_dataset_for_task(c.feature_dir,6,c.buffer_dir,SEED,start=6);ds=EffectDataset(native,6);loader=DataLoader(ds,batch_size=10,num_workers=0)
  a.start_task(-1,loader);a.refs={6:native[0][0]['feature'][0,0,0].cuda()};a.configure_effect(6,ds)
  batch=next(iter(loader));initial=capture(a);v2initial=a.export_v2_state()
  one=PaperMLR.observe(a,copy.deepcopy(batch),6,0,{'selected_pairs':[]});record('native_off_reference');native_after=capture(a)
  restore(a,initial);two=a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});record('v2_off');assert one==two and equal(native_after,capture(a))
  save(O/'preflight/recovery.pth',dict(native=capture(a),v2=a.export_v2_state()))
  a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});record('uninterrupted_second_update');after=capture(a)
  recovery=torch.load(O/'preflight/recovery.pth',map_location='cpu');restore(a,recovery['native']);a.import_v2_state(recovery['v2'])
  a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});record('restored_second_update');assert equal(after,capture(a)) and equal(v2initial,a.export_v2_state())
  # Actual next-action labels and frozen native GMM, deterministic eval mode.
  restore(a,initial);policy=a.policy.module;policy.eval();params=[p for p in policy.parameters() if p.requires_grad]
  indices=[i for i,r in enumerate(ds.rows) if r['valid']][:10];x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in indices]).cuda();actions=torch.stack([ds.rows[i]['next_action'] for i in indices]).cuda()
  base=nll(policy,x,actions).mean();bound=float(base.detach());g=torch.autograd.grad(base,params,allow_unused=True);norm=sum(float(z.double().square().sum()) for z in g if z is not None);assert norm>0
  opt=torch.optim.SGD(params,lr=.1/norm);opt.zero_grad()
  for p,z in zip(params,g):p.grad=None if z is None else -z.detach().clone()
  opt.step();record('constructed_harmful_likelihood_ascent')
  harmful=nll(policy,x,actions).mean();hv=float(harmful.detach());assert hv>bound+.05,(bound,hv)
  mu=dual({'test':0.},{'test':hv},{'test':bound})['test'];assert 0<mu<=1
  retain=mu*torch.relu(harmful-bound-.05);rg=torch.autograd.grad(retain,params,allow_unused=True);rn=sum(float(z.double().square().sum()) for z in rg if z is not None);assert rn>0 and np.isfinite(rn)
  opt=torch.optim.SGD(params,lr=.02*mu/rn);opt.zero_grad()
  for p,z in zip(params,rg):p.grad=None if z is None else z.detach().clone()
  opt.step();record('activated_retention_response');repaired=float(nll(policy,x,actions).mean().detach());assert repaired<hv,(hv,repaired)
  # Two controlled optimizer branches from the same snapshot demonstrate the
  # conflict gate cannot cancel BC, while a concordant gradient is retained.
  outcomes=[]
  for sign in (-1,1):
   restore(a,initial);policy.eval();bcx=batch[0]['feature'].cuda();bca=batch[0]['actions'].squeeze(1)[:,-1].cuda();loss=nll(policy,bcx,bca).mean()
   bg=torch.autograd.grad(loss,params,allow_unused=True);candidate=[None if z is None else sign*z for z in bg];safe,dot=safe_transfer(candidate,bg)
   assert (dot<0)==(sign<0)
   if sign<0:assert all(z is None for z in safe)
   else:assert all(s is z for s,z in zip(safe,candidate))
   n=sum(float(z.double().square().sum()) for z in bg if z is not None);opt=torch.optim.SGD(params,lr=.01/n);opt.zero_grad()
   for p,b,s in zip(params,bg,safe):p.grad=None if b is None else b.detach().clone()+(s.detach()*.05 if s is not None else 0)
   opt.step();record('plasticity_conflict_blocked' if sign<0 else 'plasticity_concordant_retained')
   improved=float(nll(policy,bcx,bca).mean().detach());assert improved<float(loss.detach())
   outcomes.append(dict(sign=sign,dot=dot,bc_before=float(loss.detach()),bc_after=improved))
  restore(a,initial);a.import_v2_state(v2initial)
 finally:dist.destroy_process_group()
 verify();assert len(ledger)==8
 write(O/'GPU_PREFLIGHT.json',dict(status='PASS',optimizer_calls=8,formal_steps=0,environment_steps=0,zero_aux_native_update_exact=True,full_state_epoch_recovery_exact=True,actual_index_wrapper_order_exact=True,retention=dict(bound=bound,harmful=hv,epsilon=.05,mu=mu,after_response=repaired,gradient_norm2=rn),plasticity=outcomes,gpu=torch.cuda.get_device_name(),torch=torch.__version__))

if __name__=='__main__':main()
