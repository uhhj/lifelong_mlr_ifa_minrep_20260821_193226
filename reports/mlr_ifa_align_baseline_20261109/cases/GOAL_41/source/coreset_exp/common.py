"""Isolated run paths and complete-state helpers; never writes parent inputs."""
import hashlib,json,os,random,copy
from pathlib import Path
import numpy as np
import torch
S=Path(__file__).resolve().parents[1]
O=Path(os.environ['CORESET_OUT'])
SEED=int(os.environ['CORESET_SEED']);assert SEED in [41,42,43]
SELECTOR_SEED=42
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def jsonable(x):
 if torch.is_tensor(x):return x.detach().cpu().tolist()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
 return x
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
def hashes(root,pattern='*.pt'):return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob(pattern))}
def source_hashes():return {str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
def verify():
 c=json.loads((O/'CONTRACT.json').read_text())
 assert source_hashes()==c['source_sha256'],'Frozen source changed'
 assert all(sha(p)==h for p,h in c['files'].items())
 return c
def capacity():
 spec=json.loads((O/'INPUT.json').read_text());base=Path(spec['all_private_outputs']);seen=set();allocated=0
 for p in base.rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  s=p.stat();key=(s.st_dev,s.st_ino)
  if key not in seen:seen.add(key);allocated+=s.st_blocks*512
 out={}
 for mount,bound,reserve in [('/data',6*1024**3,2*1024**3),('/root',128*1024**2,1024**3)]:
  s=os.statvfs(mount);free=s.f_bavail*s.f_frsize;remaining=max(0,bound-allocated) if mount=='/data' else bound
  assert free-remaining>=reserve,(mount,free,remaining,reserve)
  out[mount]=dict(free=free,remaining=remaining,reserve=reserve)
 write(O/'CAPACITY.json',out)
 return out
