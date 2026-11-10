from pathlib import Path
import sys,json,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()=='ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f'
sys.path.insert(0,str(Q))
from observe import snapshot
print(json.dumps(snapshot()))
