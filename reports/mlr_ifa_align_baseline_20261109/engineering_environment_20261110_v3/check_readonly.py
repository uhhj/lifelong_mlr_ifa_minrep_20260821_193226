from pathlib import Path
import sys,json,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()=='255748b47194a8ec8ce24485cd89399058620ab03a29c14690d8f4c652e568dd'
sys.path.insert(0,str(Q))
from observe import snapshot
print(json.dumps(snapshot()))
