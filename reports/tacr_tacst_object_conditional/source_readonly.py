import json
from pathlib import Path
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source')
names=['libero/lifelong/models/brief.py','libero/lifelong/models/complete_policy.py','libero/lifelong/algos/mlr.py','libero/lifelong/algos/basic.py','libero/lifelong/models/base_policy.py','coreset_exp/eval_metric.py']
print(json.dumps(dict(source_files={n:(S/n).read_text() for n in names if (S/n).is_file()},server_writes=0,model_forwards=0,optimizer_calls=0)))
