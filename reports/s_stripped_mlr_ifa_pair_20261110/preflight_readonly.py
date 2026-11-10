import json,os,time,subprocess,hashlib
from pathlib import Path
NAME='S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME;P=Path('/data/mlr_ifa/experiments')/NAME
OLD=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
PREV=Path('/data/mlr_ifa/experiments/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_text())
assert not Q.exists() and not P.exists(),'New root exists; do not redeploy'
assert (OLD/'engineering_cached_eval_20261110_v1/controller.exit').read_text().strip()=='0'
capacity={}
for mount in ['/','/data']:
 s=os.statvfs(mount);capacity[mount]=dict(available_bytes=s.f_bavail*s.f_frsize,total_bytes=s.f_blocks*s.f_frsize)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
cases={}
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  case=suite+'_'+str(seed);i=J(PREV/case/'INPUT.json');parent=Path(i['parent']);info={}
  for t in range(6,10):
   p=parent/('random/ifa_task%d.json'%t);info[str(t)]=dict(path=str(p),sha256=sha(p),data=J(p))
  cases[case]=dict(ifa=info,initial=J(parent/'random/INITIAL_STATE.json'),source_contract=J(parent/'CONTRACT.json'))
print(json.dumps(dict(status='READONLY_PREFLIGHT',observed=time.time(),capacity=capacity,gpu=dict(exit=gpu.returncode,output=gpu.stdout),apps=dict(exit=apps.returncode,output=apps.stdout),cases=cases,new_optimizer=0,new_environment=0)))
