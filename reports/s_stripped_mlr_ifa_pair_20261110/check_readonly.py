from pathlib import Path
import sys,json
Q=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
sys.path.insert(0,str(Q))
from runtime import snapshot
print(json.dumps(snapshot()))
