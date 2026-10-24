"""Read-only source, initialization, saved-state sizing and filesystem preflight."""
import os,json,hashlib,subprocess,time
from pathlib import Path
import torch
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1')
O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022')
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text())
assert all(sha(p)==h for p,h in c['files'].items())
assert all(sha(S/n)==h for n,h in c['parent_source'].items())
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(Q) in cmd:procs.append(dict(pid=int(p.name),cmd=cmd))
 except OSError:pass
assert not procs
cp=Path('/root/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1/B/recovery.pth')
raw=torch.load(cp,map_location='cpu');assert raw['next_epoch']==9
parts={}
def tensors(x):
 if torch.is_tensor(x):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from tensors(v)
 elif isinstance(x,(list,tuple)):
  for v in x:yield from tensors(v)
for name,value in raw['state'].items():
 parts[name]=sum(t.numel()*t.element_size() for t in tensors(value))
meta={}
for stage in [6,7,8,9]:
 man=O/('ra_coreset/dataset_manifests/task%d.json'%stage);d=json.loads(man.read_text());rows=d['rows']
 meta[str(stage)]=dict(manifest_sha256=sha(man),population=len(rows),steps_per_epoch=(len(rows)+9)//10,valid=sum(bool(r['bc_valid']) for r in rows),old_valid=sum(bool(r['bc_valid']) and r['task']<stage for r in rows),tasks=sorted(set(r['task'] for r in rows)))
cap={}
for p in ['/root','/data']:
 v=os.statvfs(p);cap[p]=dict(bavail=v.f_bavail*v.f_frsize,bfree=v.f_bfree*v.f_frsize)
sources={}
for name in ['caet_v3_exp/train.py','caet_v3_exp/model.py','coreset_exp/train.py','coreset_exp/common.py','coreset_exp/model.py','coreset_exp/pipeline.py']:
 p=S/name
 if p.exists():sources[name]=p.read_text()
small={}
for n in ['resolved_ra_coreset.json','INITIALIZATION_AUDIT.json','ra_coreset/initialization.json','ra_coreset/ifa_task6.json','ra_coreset/ifa_task7.json','ra_coreset/ifa_task8.json','ra_coreset/ifa_task9.json']:
 p=O/n
 if p.exists():small[n]=dict(text=p.read_text(),sha256=sha(p))
print(json.dumps(dict(status='READONLY_REPLACEMENT_PREFLIGHT',observed=time.time(),source_hash_pass=True,old_controller_exit=(Q/'controller.exit').read_text(),old_processes=procs,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),capacity=cap,recovery=dict(path=str(cp),bytes=cp.stat().st_size,sha256=sha(cp),next_epoch=raw['next_epoch'],steps=raw['steps'],keys=len(raw['state']['model']),tensor_bytes_by_state=parts),manifests=meta,metadata=small,sources=sources,forwards=0,updates=0,environments=0,server_writes=0)))
