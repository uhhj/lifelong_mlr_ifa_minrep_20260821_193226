"""Finite paired ablation orchestration; no scientific code in this module."""
from pathlib import Path
import json,os,time,hashlib,subprocess,uuid
NAME='S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME
P=Path('/data/mlr_ifa/experiments')/NAME
PY='/data/mlr_ifa/env/bin/python'
CASES=[s+'_'+v+'_'+str(k) for s in ['OBJECT','GOAL'] for k in [41,42,43] for v in ['ALIGN','MLR']]
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.%d'%os.getpid());tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def verify(full=False):
 c=js(Q/'CONTRACT.json');assert c['name']==NAME and c['cases']==CASES
 for rel,h in c['code_files'].items():assert sha(Q/rel)==h,rel
 if full:
  for path,h in c['external_files'].items():assert sha(path)==h,path
 return c
def capacity():
 seen=set();allocated=0
 for base in [Q,P]:
  for p in base.rglob('*') if base.exists() else []:
   if not p.is_file() or p.is_symlink():continue
   s=p.stat();key=(s.st_dev,s.st_ino)
   if key not in seen:seen.add(key);allocated+=s.st_blocks*512
 v=os.statvfs('/data');free=v.f_bavail*v.f_frsize;remaining=max(0,24*1024**3-allocated)
 assert free-remaining>=2*1024**3,('data capacity',free,remaining)
 v=os.statvfs('/');root_free=v.f_bavail*v.f_frsize;assert root_free>=1024**3+64*1024**2,('root capacity',root_free)
 return dict(data_free=free,allocated=allocated,remaining_bound=remaining,root_free=root_free)
def processes():
 out=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit() or int(d.name)==os.getpid():continue
  try:
   cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if str(Q) in cmd or str(P) in cmd:out.append(dict(pid=int(d.name),cmd=cmd,state=(d/'stat').read_text().split(') ',1)[1].split()[0]))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return out
def env(case):
 e=os.environ.copy();e.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface')
 return e
def event(kind,**kw):
 with (Q/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n');f.flush()
def snapshot():
 verify();light={}
 def add(p,rel):
  raw=p.read_bytes()
  if p.suffix=='.jsonl':raw=raw[:raw.rfind(b'\n')+1]
  light[rel]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 for n in ['controller.exit','STATUS.json','EVENTS.jsonl','FAILURE.json','DEPLOYMENT.json','OBJECT_COMPLETE.json','GOAL_COMPLETE.json','ENVIRONMENT_LOCK_EVENTS.jsonl','FINAL_CPU_REQUEST.json']:
  if (Q/n).exists():add(Q/n,'coordinator/'+n)
 excluded={'INPUT.json','SOURCE_HASHES.json','resolved_random.json','resolved_eval.json'}
 for case in CASES:
  d=Q/'cases'/case
  for p in d.iterdir():
   if p.is_file() and p.suffix in ['.json','.jsonl','.exit'] and p.name not in excluded and not p.name.startswith('ifa_task'):add(p,case+'/'+p.name)
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
 return dict(status='READONLY_PROGRESS_NOT_FINAL',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),processes=processes(),capacity=capacity(),gpu=dict(exit=gpu.returncode,output=gpu.stdout),light=light,new_optimizer=0,new_environments=0)
