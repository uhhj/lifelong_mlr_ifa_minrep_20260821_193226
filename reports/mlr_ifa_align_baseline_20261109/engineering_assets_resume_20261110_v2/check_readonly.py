from pathlib import Path
import sys,json
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(E))
from resume_runtime import *
verify()
from observe import snapshot
s=snapshot();s['engineering_contract_sha256']=sha(E/'CONTRACT.json')
for p in E.rglob('*'):
 if not p.is_file() or p.suffix not in ['.json','.jsonl','.exit']:continue
 raw=p.read_bytes()
 if p.suffix=='.jsonl':raw=raw[:raw.rfind(b'\n')+1]
 s['light']['engineering/'+str(p.relative_to(E))]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
print(json.dumps(s))
