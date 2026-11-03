import os,json,time,subprocess
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_r_goal_engineering_20261103_v1')
P=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_GOAL41_42_43_20261103_v1')
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and (str(D) in cmd or str(P) in cmd):processes.append(dict(pid=int(p.name),cmd=cmd))
 except OSError:pass
print(json.dumps(dict(status='READ_ONLY_PREWRITE_FAILURE_INSPECTION',observed=time.time(),diagnostic_directory_exists=D.exists(),production_root_exists=P.exists(),private_output_exists=(Path('/data/mlr_ifa/experiments')/P.name).exists(),processes=processes,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),optimizer_calls=0,environments=0)))
