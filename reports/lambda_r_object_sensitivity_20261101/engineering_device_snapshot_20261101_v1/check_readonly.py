"""Read-only observation of the new full-S root, including split gate records."""
import json,os,hashlib,time,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
PRIVATE=Path('/data/mlr_ifa/experiments')/Q.name
OLD=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
E=Q/'engineering_device_snapshot_20261101_v1'
ec=json.loads((E/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in ec['files'].items())
ev=json.loads((E/'VALIDATION.json').read_text());assert ev['status']=='PASS_PAIRED_FULL_EPOCH_PATH_BOTH_LAMBDAS' and ev['optimizer_completed']==ev['optimizer_started']==7828 and ev['environments']==0
assert not (Q/'controller.exit').exists(),'Never invent original exit'
engineering=dict(contract_sha256=sha(E/'CONTRACT.json'),frozen_files=len(ec['files']),completion_source='engineering_device_snapshot_20261101_v1/controller.exit',diagnostic_optimizer_calls=7828,diagnostic_extra_autograd=ev['extra_autograd'],diagnostic_environments=0)

c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
for case in [s+'_'+str(k) for s in ['L005','L020'] for k in [41,42,43]]:
 q=Q/case;cc=json.loads((q/'CONTRACT.json').read_text());assert sha(q/'CONTRACT.json')==c['children'][case]
 assert all(sha(Path(cc['source_root'])/n)==h for n,h in cc['parent_source'].items())
procs=[];oldprocs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid():
   item=dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0])
   if str(Q) in cmd:procs.append(item)
   if str(OLD) in cmd:oldprocs.append(item)
 except OSError:pass
assert not oldprocs,'Canceled mechanism must remain stopped'
light={};gate_sources={}
def add(name,raw):
 light[name]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for p in Q.rglob('*'):
 if not p.is_file() or p.suffix not in ['.json','.jsonl','.exit'] or p.name in ['CONTRACT.json','PARENT_BINDING.json','PREDEPLOY.json']:continue
 if p.name=='GATES.jsonl':
  raw=p.read_bytes();full=raw[:raw.rfind(b'\n')+1];parts={}
  for line in full.splitlines(True):
   g=json.loads(line);parts.setdefault(g['stage'],[]).append(line)
  for stage,lines in parts.items():add(p.parent.name+'/GATES_stage%d.jsonl'%stage,b''.join(lines))
  gate_sources[str(p.relative_to(Q))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),complete_bytes=len(full),partial_line_bytes=len(raw)-len(full))
 else:
  assert p.stat().st_size<8*1024**2,str(p)
  add(str(p.relative_to(Q)),p.read_bytes())
capacity={};allocated=0;seen=set()
for p in PRIVATE.rglob('*') if PRIVATE.exists() else []:
 if not p.is_file() or p.is_symlink():continue
 st=p.stat();key=(st.st_dev,st.st_ino)
 if key not in seen:seen.add(key);allocated+=st.st_blocks*512
for mount,bound,reserve,usefree in [('/root',128*1024**2,1024**3,False),('/data',12*1024**3,2*1024**3,True)]:
 v=os.statvfs(mount);free=(v.f_bfree if usefree else v.f_bavail)*v.f_frsize;remaining=max(0,bound-allocated) if mount=='/data' else bound
 capacity[mount]=dict(free=free,remaining_bound=remaining,reserve=reserve,spare=free-remaining-reserve)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(status='READ_ONLY_SNAPSHOT_NOT_FINAL',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),engineering=engineering,children=c['children'],source_files=len(c['files']),processes=procs,canceled_mechanism_processes=oldprocs,gpu_diagnostic=dict(exit=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),capacity=capacity,light=light,gate_sources=gate_sources,log_tails={str(p.relative_to(Q)):p.read_text(errors='replace')[-4000:] for p in Q.rglob('*.log')})))
