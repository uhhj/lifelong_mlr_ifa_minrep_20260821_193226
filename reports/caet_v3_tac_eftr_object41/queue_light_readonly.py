"""Small read-only follow-up after a full frozen-hash snapshot."""
import json,time,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01')
def read(p):
 if not p.exists():return None
 return json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()
d=dict(unix=time.time(),queue=read(Q/'RUN_STATUS.json'),controller_exit=read(Q/'controller.exit'),failure=read(Q/'FAILURE.json'),runs={})
for e in read(Q/'QUEUE.json'):
 o=Path(e['output']);w=Path(e['argv'][-1]);h=read(o/'ra_coreset/epochs.json') or []
 recent=[x for x in h[-10:] if x['task']==h[-1]['task']] if h else []
 stats=[]
 if h:
  last=h[-1];stats=read(o/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(last['task'],last['epoch']))) or []
 d['runs'][e['key']]=dict(status=read(o/'RUN_STATUS.json'),complete_epochs=len(h),last=h[-1] if h else None,recent_epoch_mean_seconds=sum(x['seconds'] for x in recent)/len(recent) if recent else None,gate=read(o/'GPU_PREFLIGHT.json'),task6=read(o/'TASK6_TENSOR_PARITY.json'),supervisor_exit=read(w/'supervisor.exit'),failure=read(w/'FAILURE.json'),teacher_metadata={p.name:read(p) for p in (o/'teachers').glob('*.json')},last_epoch_mechanism=dict(records=len(stats),optimizer_trials=sum(x['optimizer_trials'] for x in stats),selected={str(i):sum(x.get('selected')==i for x in stats) for i in range(3)},budget_unmet=sum(x.get('budget_unmet',False) for x in stats)),formal_records=sum(len(p.read_text().splitlines()) for p in (o/'ra_coreset/evaluation').glob('trials*.jsonl')))
print(json.dumps(d))
