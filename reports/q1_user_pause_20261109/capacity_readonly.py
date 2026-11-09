import json,os
from pathlib import Path
p=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1/user_pause_20261109_v1/STATUS.json')
d=json.loads(p.read_text())
live={}
for pid,x in d['after'].items():
 q=Path('/proc')/pid
 if q.exists():
  s=(q/'stat').read_text().split(') ',1)[1].split();live[pid]=dict(state=s[0],start=s[19],same_start=s[19]==x['start'])
 else:live[pid]=dict(missing=True)
print(json.dumps(dict(pause_receipt=d,actual_processes=live)))
