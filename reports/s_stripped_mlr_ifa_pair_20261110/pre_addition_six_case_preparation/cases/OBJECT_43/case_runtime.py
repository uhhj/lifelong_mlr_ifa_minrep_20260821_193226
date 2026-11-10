"""Complete-S runtime with frozen Random membership and ALIGN-only targets."""
import os,sys,json,hashlib,time,copy,pickle
from pathlib import Path
R=Path(__file__).resolve().parent
def js(p):return json.loads(Path(p).read_text())
INPUT=js(R/'INPUT.json')
O=Path(INPUT['align_parent']);S=Path(INPUT['align_source'])
from layout import *
os.environ.update(CORESET_OUT=str(O),CORESET_SEED=str(SEED),CAET_VARIANT='ALIGN',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',MUJOCO_GL='egl',MPLBACKEND='Agg',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(S))
import numpy as np
import torch
from easydict import EasyDict
from coreset_exp.common import sha,write,save,rng,restore_rng
from caet_v3_exp.train import cpu,build
from caet_v3_exp.model import hidden,masked_bc
from coreset_exp.model import angular_mean
from libero.lifelong.algos.mlr import FileTrajectoryDataset
from libero.lifelong.utils import control_seed
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
CALLS={'production':0,'admission':0}
COUNT_MODE='admission'
_STEP=torch.optim.AdamW.step
def counted_step(self,*args,**kw):
 result=_STEP(self,*args,**kw);CALLS[COUNT_MODE]+=1;return result
torch.optim.AdamW.step=counted_step
def set_count_mode(mode):
 global COUNT_MODE
 assert mode in CALLS;COUNT_MODE=mode
def verify():
 q=R.parents[1];c=js(q/'CONTRACT.json')
 for p,h in c['code_files'].items():assert sha(q/p)==h,p
 for p,h in js(R/'SOURCE_HASHES.json').items():assert sha(p)==h,p
 return sha(q/'CONTRACT.json')
def capacity():
 # Shared whole-experiment allocation; no duplicated per-process reservation.
 allocated=0;seen=set()
 for base in [DATA_PRIVATE,WORKSPACE]:
  for p in base.rglob('*') if base.exists() else []:
   if not p.is_file() or p.is_symlink():continue
   s=p.stat();key=(s.st_dev,s.st_ino)
   if key not in seen:seen.add(key);allocated+=s.st_blocks*512
 v=os.statvfs('/data');free=v.f_bavail*v.f_frsize;remaining=max(0,12*1024**3-allocated)
 assert free-remaining>=2*1024**3,('data capacity',free,remaining)
 v=os.statvfs('/');root_free=v.f_bavail*v.f_frsize;assert root_free>=1024**3+64*1024**2,('root capacity',root_free)
 return dict(data_free=free,remaining_bound=remaining,root_free=root_free)
class Dataset(torch.utils.data.Dataset):
 def __init__(self,stage):
  self.stage=stage;m=js(R/'manifests'/('task%d.json'%stage));self.man=m['rows'];assert len(self.man)==POP[stage]
  parent=Path(INPUT['parent']);paths=[str(parent/r['relative_path']) for r in self.man]
  self.native=FileTrajectoryDataset(paths);labels={}
  for t in range(6,stage+1):
   meta=INPUT['labels'][str(t)];p=Path(meta['path']);assert sha(p)==meta['sha256']
   labels[t]=torch.load(p,map_location='cpu')
  self.rows=[labels[r['task']][r['sample']] for r in self.man]
  for p,r,v in zip(paths,self.man,self.rows):
   assert sha(p)==r['sha256']==v['sample_sha256'],p
   assert all(r[k]==v[k] for k in ['task','demo','u','bc_valid']),p
  assert sum(bool(r['bc_valid']) for r in self.rows)==VALID[stage]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x,t=self.native[i];r=self.rows[i]
  x.update(next_action=torch.as_tensor(r['next_action'],dtype=torch.float32),causal_bc_valid=bool(r['bc_valid']),v3_index=i)
  return x,t
def make(arm):
 c=EasyDict(js(R/'resolved_random.json'));c.experiment_dir=str(OUTPUT[arm]/'models');c.run_root=str(OUTPUT[arm]);c.buffer_dir=str(OUTPUT[arm]/'logical_buffer_empty')
 a,b=build(c);a.refs={};a.caet_epoch_statistics=[];a.disable_all=False
 assert a.step==0
 return a
def setup(a,ds):
 sampler=torch.utils.data.DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0)
 loader=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0,persistent_workers=False)
 a.start_task(-1,loader);a.refs={}
 for t in range(6,ds.stage+1):
  i=next(i for i,r in enumerate(ds.rows) if r['task']==t)
  f=ds.native[i][0]['feature'][0,:,0,:];a.refs[t]=f[0].cuda()
 # Read the exact LambdaLR closure instead of assuming library attributes.
 fn=a.scheduler.lr_lambdas[0];closure=dict(zip(fn.__code__.co_freevars,[x.cell_contents for x in (fn.__closure__ or [])]))
 steps=100*len(loader)
 assert a.cfg.train.n_epochs==100 and a.cfg.train.batch_size==10 and a.cfg.train.scheduler.kwargs.warmup_steps==500
 assert any(v==steps for v in closure.values()),('schedule total steps',steps,closure)
 write(R/('SCHEDULE_A_%d.json'%ds.stage),dict(status='PASS',stage=ds.stage,total_updates=steps,warmup=500,closure={k:v for k,v in closure.items() if isinstance(v,(int,float,str,bool))}))
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
def initialization(a,arm):
 r=rng();digest={k:hashlib.sha256(pickle.dumps(v) if k in ('python','numpy') else b''.join(x.cpu().numpy().tobytes() for x in (v if isinstance(v,list) else [v]))).hexdigest() for k,v in r.items()}
 tensor={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in a.policy.state_dict().items()}
 expected=INPUT['initial_state'];assert tensor==expected['tensors'] and all(digest[k]==expected['rng'][k] for k in digest if k!='cuda'),'Official initialization/non-CUDA RNG mismatch'
 props=torch.cuda.get_device_properties(0)
 hardware=dict(device=props.name,capability=[props.major,props.minor],total_memory=props.total_memory,torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
 write(R/('INITIAL_'+arm+'.json'),dict(status='PASS_OFFICIAL_INITIALIZATION',tensors=tensor,rng=digest,hardware=hardware,historical_cuda_rng_equal=digest['cuda']==expected['rng']['cuda'],historical_cuda_rng_gate=False,official_cp_sha256=sha(a.cfg.pretrain)))
