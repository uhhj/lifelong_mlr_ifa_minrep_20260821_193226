from pathlib import Path
import sys,json,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1')
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()=='db88ef7991d4cfd6e77b6897977779ed5d37a85682cd63c14441bb63bddcd7c3'
sys.path.insert(0,str(Q))
from observe import snapshot
print(json.dumps(snapshot()))
