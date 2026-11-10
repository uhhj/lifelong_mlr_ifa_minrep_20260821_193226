"""One-shot isolated deployment. No old process/file mutations."""
from pathlib import Path
import json,hashlib,os,subprocess,time
NAME='S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME;P=Path('/data/mlr_ifa/experiments')/NAME
PY='/data/mlr_ifa/env/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not Q.exists() and not P.exists(),'Unique deployment already exists; do not repeat'
c=json.loads(BUNDLE['CONTRACT.json'])
for rel,h in c['code_files'].items():assert hashlib.sha256(BUNDLE[rel].encode()).hexdigest()==h,rel
for n,s in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts
 if n.endswith('.py'):compile(s,n,'exec')
for path,h in c['external_files'].items():assert sha(path)==h,('Immutable input changed',path)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU must be idle before unique launch'
v=os.statvfs('/data');df=v.f_bavail*v.f_frsize;assert df>=26*1024**3,('data capacity',df)
v=os.statvfs('/');rf=v.f_bavail*v.f_frsize;assert rf>=1024**3+64*1024**2,('root capacity',rf)
# Previous six-run science really exited. No need to restart or modify it.
old=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
assert (old/'engineering_cached_eval_20261110_v1/controller.exit').read_text().strip()=='0'
Q.mkdir();P.mkdir()
for n,s in BUNDLE.items():
 dest=Q/n;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('x',encoding='utf-8',newline='\n') as f:f.write(s)
for case in c['cases']:(P/case/'A').mkdir(parents=True)
with (Q/'controller.log').open('x') as log:
 proc=subprocess.Popen([PY,'-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='DEPLOYED_ONCE_PENDING_ACTUAL_ADMISSION',controller=proc.pid,root=str(Q),private_root=str(P),contract_sha256=sha(Q/'CONTRACT.json'),observed=time.time(),capacity=dict(data_free=df,root_free=rf),external_files_verified=len(c['external_files']),training_collection=0)
(Q/'DEPLOYMENT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
