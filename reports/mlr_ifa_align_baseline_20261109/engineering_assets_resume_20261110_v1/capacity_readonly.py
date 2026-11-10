from pathlib import Path
import sys,json,time,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v1'
sys.path.insert(0,str(E));from resume_runtime import *
verify();files={}
for p in [E/'OBJECT_41_evaluate.log',E/'FAILURE.json',E/'controller.exit',E/'RNG_BOUNDARY_AUDIT.json']+list((P/'OBJECT_41/random/evaluation').glob('*')):
 if p.is_file():
  x=dict(sha256=sha(p),bytes=p.stat().st_size)
  if p.suffix in ['.json','.jsonl','.log','.exit']:x['text']=p.read_text()[-20000:]
  files[str(p)]=x
print(json.dumps(dict(status='READONLY_EXISTING_RECOVERY_FAILURE',files=files,processes=processes([Q,P]),observed=time.time(),new_optimizer=0,new_environment=0)))
