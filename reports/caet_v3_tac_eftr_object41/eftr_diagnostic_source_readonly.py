"""Read exact active model constructors and execution logic, without imports."""
import json, hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01')
S=Q/'EFTR/source'
O=Path('/data/mlr_ifa/experiments/CAETv3_EFTR_OBJECT_seed41_20261022_collate01')
files={}
for name in ['libero/lifelong/models/modules/transformer_modules.py']:
 p=S/name; raw=p.read_bytes();files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
print(json.dumps({'files':files,'worker_files':[p.name for p in (Q/'EFTR').iterdir()], 'recovery_bytes':(O/'ra_coreset/recovery/latest.pth').stat().st_size,'status':json.loads((O/'RUN_STATUS.json').read_text())}))
