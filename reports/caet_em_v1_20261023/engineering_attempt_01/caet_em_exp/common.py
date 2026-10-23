import os,sys,json,hashlib,random,copy,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ.get('EM_ROOT','/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1'))
PARENT=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
SOURCE=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
sys.path.append(str(SOURCE))
os.environ['CORESET_SEED']='41';os.environ['CORESET_OUT']=str(ROOT)
torch.set_num_threads(1)
ARMS=('B0','B1','B2');POPS={6:1562,7:2261,8:3081,9:3620};BATCHES={6:157,7:227,8:309,9:362};QUOTAS={6:781,7:740,8:780,9:660}
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def event(kind,**kw):append(ROOT/'EVENTS.jsonl',dict(event=kind,unix=time.time(),pid=os.getpid(),**kw))
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['cpu']);torch.cuda.set_rng_state_all(x['cuda'])
def seed(s=41):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def config():
 from easydict import EasyDict
 c=EasyDict(js(PARENT/'resolved_eval.json'));assert c.seed==41 and c.gamma==.3 and c.train.batch_size==10
 return c
def shape():return {'all_shapes':{'agentview_rgb':(3,128,128),'eye_in_hand_rgb':(3,128,128),'joint_states':(7,),'gripper_states':(2,)},'ac_dim':7}
def official():
 raw=torch.load(config().pretrain,map_location='cpu')['state_dict']
 return {k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in raw.items()}
def arm_root(a):assert a in ARMS;return Path(js(ROOT/'ENTRY.json')['arms'][a])
def capacity():
 d={p:dict(root_free=os.statvfs(p).f_bfree*os.statvfs(p).f_frsize,ordinary_free=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize) for p in ('/data','/')}
 assert d['/data']['root_free']>=2*1024**3 and d['/']['ordinary_free']>=1024**3,d
 return d
def verify():
 d=js(ROOT/'CONTRACT.json');bad=[p for p,h in d['files'].items() if sha(p)!=h];assert not bad,bad
 return sha(ROOT/'CONTRACT.json')
def frozen_codec():
 p=ROOT/'shared/codec_frozen.pth';m=js(ROOT/'CODEC_FREEZE_MANIFEST.json');assert sha(p)==m['sha256'];return p
def full_frontend(device='cuda'):
 from libero.lifelong.models.complete_policy import CompletePolicy
 c=config();c.shape_meta=shape();m=CompletePolicy(c,c.shape_meta);raw=official();m.load_state_dict(raw,strict=True)
 assert set(raw)==set(m.state_dict()) and all(torch.equal(v,m.state_dict()[k]) for k,v in raw.items())
 return m.to(device).eval().requires_grad_(False)
def capture(model,opt,sched,**kw):
 return dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(sched.state_dict()) if sched else None,rng=cpu(rng()),grads={n:cpu(p.grad) for n,p in model.named_parameters()},modes={n:m.training for n,m in model.named_modules()},requires={n:p.requires_grad for n,p in model.named_parameters()},**kw)
def restore(model,opt,sched,x):
 model.load_state_dict(x['model'],strict=True);opt.load_state_dict(x['optimizer'])
 if sched:sched.load_state_dict(x['scheduler'])
 for n,p in model.named_parameters():p.requires_grad_(x['requires'][n]);p.grad=None if x['grads'][n] is None else x['grads'][n].to(p.device)
 for n,m in model.named_modules():m.training=x['modes'][n]
 restore_rng(x['rng'])
def to_device(x,device='cuda'):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
