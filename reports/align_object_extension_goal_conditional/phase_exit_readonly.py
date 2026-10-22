import json,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022')
out=dict(observed_unix=time.time(),runs={},server_writes=0,optimizer_calls=0,environment_steps=0)
for e in json.loads((Q/'QUEUE.json').read_text()):
 O=Path(e['output'])
 if not O.exists():continue
 out['runs'][e['key']]=dict(phase_exits={p.name:p.read_text() for p in O.glob('*.exit')},status=json.loads((O/'RUN_STATUS.json').read_text()))
 assert all(x.strip()=='0' for x in out['runs'][e['key']]['phase_exits'].values())
print(json.dumps(out))
