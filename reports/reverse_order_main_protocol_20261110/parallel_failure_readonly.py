from pathlib import Path
import json,time
root=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
case=root/'cases/GOAL_MLR_41'
paths=[root/'parallel_dispatch_v1.log',case/'train.log',case/'train.exit',case/'OPTIMIZER_CALLS.json',root/'STATUS.json']
values={}
for path in paths:
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-16000));values[path.name]=f.read().decode(errors='replace')
print(json.dumps(dict(status='READONLY_FAILURE_EVIDENCE',observed=time.time(),logs=values,new_optimizer=0,new_environments=0)))
