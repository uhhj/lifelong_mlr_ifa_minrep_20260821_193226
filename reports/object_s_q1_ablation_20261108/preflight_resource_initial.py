"""Read-only resource, uniqueness and baseline availability preflight."""
import os,json,time,subprocess,hashlib
from pathlib import Path
name='S_Q1_OBJECT41_42_43_20261108_v1'
roots={k:Path(v) for k,v in dict(new='/data/mlr_ifa/experiment_workspaces/'+name,baseline='/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1').items()}
def cmd(argv):
 p=subprocess.run(argv,capture_output=True,text=True);return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
capacity={}
for mount,bound,reserve in [('/root',128*1024**2,1024**3),('/data',8*1024**3,2*1024**3)]:
 s=os.statvfs(mount);free=s.f_bavail*s.f_frsize
 capacity[mount]=dict(available=free,provisional_peak=bound,reserve=reserve,headroom=free-bound-reserve)
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and ('mlr_ifa' in c or 'torch.distributed' in c):processes.append(dict(pid=int(p.name),command=c))
 except OSError:pass
base=roots['baseline'];files={}
if base.exists():
 for p in base.rglob('*.exit'):files[str(p.relative_to(base))]=dict(text=p.read_text(),sha256=sha(p))
 p=base/'CONTRACT.json'
 if p.exists():files['CONTRACT.json']=dict(sha256=sha(p))
print(json.dumps(dict(status='READ_ONLY_RESOURCE_OBSERVATION',observed=time.time(),roots={k:dict(path=str(p),exists=p.exists()) for k,p in roots.items()},output_exists=Path('/data/mlr_ifa/experiments',name).exists(),capacity=capacity,processes=processes,hardware=cmd(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.free,driver_version','--format=csv,noheader']),gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),baseline=files,optimizer_calls=0,environments=0)))
