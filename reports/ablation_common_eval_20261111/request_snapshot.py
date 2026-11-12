from pathlib import Path
import json,subprocess,sys
root=Path('/data/mlr_ifa/experiment_workspaces/ABLATION_COMMON_DEBR_EVAL_20261111_v2')
sys.path.insert(0,str(root))
from job import snapshot
print(json.dumps(snapshot()))
