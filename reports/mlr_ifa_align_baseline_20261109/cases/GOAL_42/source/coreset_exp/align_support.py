"""Only ALIGN targets/masks plus isolated admission and traceability."""
import copy,hashlib,io,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from .common import O,SEED,sha,write,rng,restore_rng,capacity
CALLS=dict(production_started=0,production_completed=0,admission_started=0,admission_completed=0)
MODE='admission'
ORIGINAL=torch.optim.AdamW.step
def counted(self,*a,**kw):
 CALLS[MODE+'_started']+=1
 if MODE=='admission':write(O/'OPTIMIZER_CALLS.json',CALLS)
 out=ORIGINAL(self,*a,**kw);CALLS[MODE+'_completed']+=1
 if MODE=='admission':write(O/'OPTIMIZER_CALLS.json',CALLS)
 return out
torch.optim.AdamW.step=counted
def mode(value):
 global MODE
 assert value in ['admission','production'];MODE=value
def aligned_bc(policy,data):
 x=data['feature']
 if x.ndim==5:x=x.squeeze(1)
 assert x.shape[1:]==(8,4,512)
 h=policy.temporal_encode(x)[:,-1]
 gmm=policy.policy_head(h);valid=data['causal_bc_valid'].bool()
 values=-gmm.log_prob(data['next_action'])*policy.policy_head.loss_coef
 loss=values[valid].mean() if valid.any() else values.sum()*0
 return loss,h
class AlignedDataset(torch.utils.data.Dataset):
 def __init__(self,native,stage):
  self.native=native;self.file_paths=native.file_paths;self.stage=stage
  spec=json.loads((O/'INPUT.json').read_text());manifest=json.loads((O/'manifests'/('task%d.json'%stage)).read_text())
  actual=[str(Path(p).relative_to(O)) for p in self.file_paths]
  assert actual==[r['relative_path'] for r in manifest['rows']]
  labels={}
  for t in range(6,stage+1):
   m=spec['labels'][str(t)];assert sha(m['path'])==m['sha256']
   labels[t]=torch.load(m['path'],map_location='cpu')
  self.rows=[]
  for p,r in zip(self.file_paths,manifest['rows']):
   label=labels[r['task']][r['sample']]
   assert sha(p)==r['sha256']==label['sample_sha256']
   assert bool(label['bc_valid'])==r['bc_valid']
   for k in ['demo','u']:
    if r.get(k) is not None:assert label[k]==r[k]
   self.rows.append(label)
  self.manifest=manifest
 def __len__(self):return len(self.native)
 def __getitem__(self,i):
  x,t=self.native[i];r=self.rows[i]
  x.update(next_action=torch.as_tensor(r['next_action'],dtype=torch.float32),causal_bc_valid=bool(r['bc_valid']))
  return x,t
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def initialize(a,evidence):
 spec=json.loads((O/'INPUT.json').read_text());expected=spec['initial_state']
 assert evidence==expected,'Original Random initial tensors/RNG mismatch'
 write(O/'INITIALIZATION_AUDIT.json',dict(status='PASS_ACTUAL_INITIALIZATION',initial_exact=True,device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,baseline_gpu=spec['baseline_gpu'],hardware_controlled=False))
def admit(a,ds,info,capture,restore):
 from .reference_train import PaperMLR as OriginalPaperMLR
 capacity();start=capture(a);old_calls=dict(CALLS);stage=ds.stage
 batch=torch.utils.data.default_collate([ds[i] for i in range(min(10,len(ds)))])
 try:
  mode('admission')
  OriginalPaperMLR.observe(a,copy.deepcopy(batch),stage,0,info);native=capture(a)
  restore(a,start);a.disable_align=True;a.observe(copy.deepcopy(batch),stage,0,info);assert equal(native,capture(a));a.disable_align=False
  restore(a,start);x=a.map_tensor_to_device(copy.deepcopy(batch[0]));loss,h=aligned_bc(a.policy.module,x)
  gmm=a.policy.module.policy_head(h);valid=x['causal_bc_valid'].bool()
  expected=(-gmm.log_prob(x['next_action'])*a.policy.module.policy_head.loss_coef)[valid].mean() if valid.any() else loss*0
  assert torch.equal(loss,expected)
  xx=dict(x,causal_bc_valid=torch.zeros_like(valid));zero,_=aligned_bc(a.policy.module,xx);assert float(zero)==0 and zero.requires_grad
  restore(a,start);a.observe(copy.deepcopy(batch),stage,0,info);middle=capture(a)
  stream=io.BytesIO();torch.save(middle,stream);stream.seek(0)
  a.observe(copy.deepcopy(batch),stage,0,info);after=capture(a)
  restore(a,torch.load(stream,map_location='cpu'));a.observe(copy.deepcopy(batch),stage,0,info);assert equal(after,capture(a))
  assert CALLS['admission_completed']-old_calls['admission_completed']==5 and CALLS['admission_started']==CALLS['admission_completed']
  restore(a,start);assert equal(start,capture(a))
  write(O/('ADMISSION_%d.json'%stage),dict(status='PASS_OWN_GPU_ALIGN_ADMISSION',stage=stage,optimizer_calls=5,disabled_original_random_full_state_exact=True,valid_mask_nll_exact=True,empty_mask_differentiable_zero=True,serialized_full_state_next_step_exact=True,initial_state_restored=True,production_calls_added=0,environments=0))
 finally:
  a.disable_align=False;restore(a,start);mode('production')
def record_epoch(a,stage,epoch,ds,sampler):
 write(O/'OPTIMIZER_CALLS.json',CALLS)
 assert CALLS['production_started']==CALLS['production_completed']==a.step
 ids=list(iter(sampler));write(O/('SAMPLER_%d.json'%stage),dict(stage=stage,latest_epoch=epoch,indices_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),population=len(ds),bc_valid=sum(r['bc_valid'] for r in ds.rows)))
 if epoch==0 or epoch==99:capacity()
