import os,sys,json,hashlib,time,copy
from pathlib import Path
R=Path(__file__).resolve().parent
O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed43_20261022')
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_43/source')
from layout import *
os.environ.update(CORESET_OUT=str(O),CORESET_SEED='43',CAET_VARIANT='ALIGN',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',MUJOCO_GL='egl',MPLBACKEND='Agg',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(S))
import numpy as np
import torch
from easydict import EasyDict
from coreset_exp.common import sha,write,save,rng,restore_rng
from caet_v3_exp.train import cpu,build
from caet_v3_exp.model import hidden,masked_bc,params
from coreset_exp.model import angular_mean
from libero.lifelong.algos.mlr import FileTrajectoryDataset
from libero.lifelong.utils import control_seed
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
def js(p):return json.loads(Path(p).read_text())
def verify():
 contract=js(R/'CONTRACT.json')
 for path,h in contract['files'].items():assert sha(path)==h,path
 pc=js(O/'CONTRACT.json');assert all(sha(S/n)==h for n,h in pc['source_sha256'].items())
 return sha(R/'CONTRACT.json')
def capacity():
 out={}
 for mount,base,reserve,root_free in [('/root',ROOT_PRIVATE,1024**3,False),('/data',DATA_PRIVATE,2*1024**3,True)]:
  seen=set();allocated=0
  for p in base.rglob('*') if base.exists() else []:
   if not p.is_file() or p.is_symlink():continue
   st=p.stat();key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);allocated+=st.st_blocks*512
  v=os.statvfs(mount);free=(v.f_bfree if root_free else v.f_bavail)*v.f_frsize
  remaining=max(0,VOLUME_BOUND[mount]-allocated);assert free-remaining>=reserve,(mount,free,remaining)
  out[mount]=dict(free=free,unique_inode_allocated=allocated,remaining_bound=remaining,reserve=reserve)
 return out
class Dataset(torch.utils.data.Dataset):
 def __init__(self,stage):
  self.stage=stage;self.man=js(O/('ra_coreset/dataset_manifests/task%d.json'%stage))['rows'];assert len(self.man)==POP[stage]
  self.native=FileTrajectoryDataset([r['path'] for r in self.man]);labels={}
  for t in range(6,stage):
   p=O/('effects/memory/task%d.pth'%t);assert sha(p)==js(p.with_suffix('.json'))['sha256']
   labels.update({str(O/'ra_coreset/buffer'/n):v for n,v in torch.load(p,map_location='cpu').items()})
  p=O/('effects/current/task%d/labels.pth'%stage);assert sha(p)==js(p.parent/'AUDIT.json')['labels_sha256']
  labels.update({str(O/'features'/str(stage)/n):v for n,v in torch.load(p,map_location='cpu').items()})
  self.rows=[labels[r['path']] for r in self.man]
  for r,v in zip(self.man,self.rows):
   assert sha(r['path'])==r['sha256']==v['sample_sha256']
   assert all(r[k]==v[k] for k in ['task','demo','u','bc_valid'])
  self.ids=[i for i,r in enumerate(self.rows) if r['task']<stage and r['bc_valid']]
  assert sum(r['bc_valid'] for r in self.rows)==VALID[stage] and len(self.ids)==OLD[stage]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x,t=self.native[i];r=self.rows[i]
  x.update(next_action=torch.as_tensor(r['next_action'],dtype=torch.float32),causal_bc_valid=bool(r['bc_valid']),v3_index=i)
  return x,t
def make(arm):
 c=EasyDict(js(O/'resolved_ra_coreset.json'));c.experiment_dir=str(OUTPUT[arm]/'models');c.run_root=str(OUTPUT[arm]);c.buffer_dir=str(OUTPUT[arm]/'logical_buffer_empty')
 a,b=build(c);a.refs={};a.caet_epoch_statistics=[];a.disable_all=False
 assert a.step==0
 return a
def setup(a,ds):
 sampler=torch.utils.data.DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0)
 loader=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0,persistent_workers=False)
 a.start_task(-1,loader);a.refs={}
 assert a.cfg.train.n_epochs==100 and not hasattr(a,'total_steps')
 fn=a.scheduler.lr_lambdas[0];assert fn(0)==0 and fn(500)==1 and fn((100*len(loader)))==0 and fn((100*len(loader))-1)>0
 write(R/('SCHEDULE_S_%d.json'%ds.stage),dict(status='PASS_ACTUAL_SCHEDULE',epochs=100,batches=len(loader),total_steps=(100*len(loader)),warmup_steps=500))
 for t in range(6,ds.stage+1):
  i=next(i for i,r in enumerate(ds.rows) if r['task']==t)
  f=ds.native[i][0]['feature'][0,:,0,:];a.refs[t]=f[0].cuda()
 return loader,sampler
def capture(a):
 return dict(model=cpu(a.policy.state_dict()),optimizer=cpu(a.optimizer.state_dict()),scheduler=cpu(a.scheduler.state_dict()),grads={n:cpu(p.grad) if p.grad is not None else None for n,p in a.policy.named_parameters()},modes={n:m.training for n,m in a.policy.named_modules()},requires={n:p.requires_grad for n,p in a.policy.named_parameters()},rng=cpu(rng()),step=a.step)
def restore(a,x):
 a.policy.load_state_dict(x['model'],strict=True);a.optimizer.load_state_dict(x['optimizer']);a.scheduler.load_state_dict(x['scheduler']);a.step=x['step']
 for n,p in a.policy.named_parameters():p.grad=None if x['grads'][n] is None else x['grads'][n].to(p.device);p.requires_grad_(x['requires'][n])
 for n,m in a.policy.named_modules():m.training=x['modes'][n]
 restore_rng(x['rng'])
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b


# Same tested copy optimization, verified at every actual stage start.
from candidate import cpu_snapshot
_reference_cpu=cpu
_reference_capture=capture
_copy_checked={}
cpu=cpu_snapshot
def capture(a):
 result=_reference_capture(a)
 if a.scheduler.last_epoch==0 and str(a.step) not in _copy_checked:
  global cpu
  try:
   cpu=_reference_cpu;oracle=_reference_capture(a)
  finally:cpu=cpu_snapshot
  assert equal(result,oracle)
  _copy_checked[str(a.step)]=dict(all_state_exact=True,optimizer_calls_added=0,environments_added=0)
  write(R/'COPY_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=_copy_checked,candidate_sha256=sha(R/'candidate.py')))
 return result
import functools
CALLS={'production':0,'admission':0}
_count_mode='admission'
_original_adamw_step=torch.optim.AdamW.step
@functools.wraps(_original_adamw_step)
def _counted_step(self,*args,**kwargs):
 result=_original_adamw_step(self,*args,**kwargs);CALLS[_count_mode]+=1;return result
torch.optim.AdamW.step=_counted_step
def set_count_mode(mode):
 global _count_mode
 assert mode in CALLS;_count_mode=mode
