from pathlib import Path
import sys,json,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1')
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()=='6a6298283056986941016d1f487f8169fcbfe760679088e675f51ceaa015f7d7'
sys.path.insert(0,str(Q))
from observe import snapshot
print(json.dumps(snapshot()))
