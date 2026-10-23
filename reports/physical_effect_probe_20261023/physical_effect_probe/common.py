import os,sys,json,hashlib,random,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(os.environ.get('PEP_ROOT','/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023'))
OUT=ROOT/'output'
SOURCE=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
PARENT=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
sys.path.insert(0,str(SOURCE))
torch.set_num_threads(1)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(str(tmp),str(p))
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(d,tmp);os.replace(str(tmp),str(p))
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def capacity(need=0):
 s=os.statvfs('/data');free=s.f_bfree*s.f_frsize;assert free>=2*1024**3+need,('data reserve',free,need)
 s=os.statvfs('/');assert s.f_bavail*s.f_frsize>=1024**3
 return free
def log(event,**kw):
 p=OUT/'EVENTS.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(event=event,unix=time.time(),**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def config():
 from easydict import EasyDict
 c=EasyDict(js(PARENT/'resolved_eval.json'));assert c.seed==41 and c.gamma==.3
 return c
def shape():
 return {'all_shapes':{'agentview_rgb':(3,128,128),'eye_in_hand_rgb':(3,128,128),'joint_states':(7,),'gripper_states':(2,)},'ac_dim':7}
def official_state():
 raw=torch.load(config().pretrain,map_location='cpu')['state_dict']
 assert all(k.startswith('module.') for k in raw)
 return {k[len('module.'):]:v for k,v in raw.items()}
def temporal(device='cuda'):
 from libero.lifelong.models.modules.transformer_modules import OfficialGPT2Decoder
 m=OfficialGPT2Decoder(512,6,8,.15,32,train=True)
 raw=official_state();prefix='temporal_transformer.'
 sd={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)};m.load_state_dict(sd,strict=True)
 m.eval().requires_grad_(False);return m.to(device)
def full_policy(device='cuda'):
 from libero.lifelong.models.complete_policy import CompletePolicy
 c=config();c.shape_meta=shape();p=CompletePolicy(c,c.shape_meta)
 raw=official_state();p.load_state_dict(raw,strict=True)
 assert set(raw)==set(p.state_dict()) and all(torch.equal(v,p.state_dict()[k]) for k,v in raw.items())
 return p.to(device).eval().requires_grad_(False)
def verify():
 d=js(ROOT/'CONTRACT.json')
 bad=[p for p,h in d['files'].items() if sha(p)!=h];assert not bad, bad
 return sha(ROOT/'CONTRACT.json')
