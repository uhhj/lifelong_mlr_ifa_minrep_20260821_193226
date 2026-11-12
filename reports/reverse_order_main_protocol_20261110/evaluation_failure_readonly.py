from pathlib import Path
import json,time
r=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
c=r/'cases/OBJECT_MLR_41'
logs={}
for name in ('evaluate.log','EVAL_EVENTS.jsonl','EVAL_ADMISSION.json','EVALUATION_STARTED.json','EVALUATION_PROTOCOL_DECLARATION.json','EVAL_DISPATCH.json'):
 p=c/name
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-18000));logs[name]=f.read().decode(errors='replace')
cases={}
for x in sorted((r/'cases').iterdir()):
 if not x.is_dir():continue
 meta=json.loads((x/'INPUT.json').read_text());f=Path(meta['output_root'])/'epochs.json' if meta['method']=='MLR' else x/'EPOCHS_S.json'
 cases[x.name]=dict(train_exit=(x/'train.exit').read_text().strip() if (x/'train.exit').exists() else None,epochs=len(json.loads(f.read_text())) if f.exists() else 0,formal=len(json.loads((x/'OUTCOMES.json').read_text())) if (x/'OUTCOMES.json').exists() else 0,eval_started=(x/'EVALUATION_STARTED.json').exists(),eval_events=(x/'EVAL_EVENTS.jsonl').exists(),outcome_shards=len(list((x/'outcomes').glob('*.json'))))
print(json.dumps(dict(status='READONLY_EVAL_FAILURE_EVIDENCE',observed=time.time(),cases=cases,logs=logs,new_optimizer=0,new_environments=0)))
