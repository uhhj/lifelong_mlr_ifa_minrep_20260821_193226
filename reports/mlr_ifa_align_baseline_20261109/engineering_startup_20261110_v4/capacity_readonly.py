"""Preserve and inspect existing failure evidence; no environment/model execution."""
from pathlib import Path
import sys,json,hashlib,subprocess,time
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
P=Path('/data/mlr_ifa/experiments')/Q.name
sys.path.insert(0,str(Q))
from runtime import verify,processes
verify()
files={}
for p in [Q/'FAILURE.json',Q/'controller.exit',Q/'controller.log',P/'OBJECT_41/evaluate.log',P/'OBJECT_41/evaluate.exit']+list((P/'OBJECT_41/random/evaluation').glob('*')):
 if p.is_file() and p.suffix in ['.log','.exit','.json','.jsonl']:
  raw=p.read_bytes();files[str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode(errors='replace')[-30000:])
r=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(status='READONLY_FAILURE_DIAGNOSIS_NOT_RECOVERY',observed=time.time(),files=files,processes=processes([Q,P]),gpu=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),new_optimizer=0,new_environments=0)))
