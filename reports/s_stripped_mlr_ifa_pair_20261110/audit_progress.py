"""Recompute recorded counters and immutable prefixes locally; no science."""
from pathlib import Path
import sys,json,hashlib,datetime
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def audit(directory,previous=None):
 d=Path(directory);e=read(d/'EVIDENCE.json');c=read(R/'CONTRACT.json')
 assert e['contract_sha256']==hashlib.sha256((R/'CONTRACT.json').read_bytes()).hexdigest()
 light=e['light'];obj={}
 for name,row in light.items():
  assert hashlib.sha256(row['text'].encode()).hexdigest()==row['sha256'],name
  if name.endswith('.json'):obj[name]=json.loads(row['text'])
  elif name.endswith('.jsonl'):obj[name]=[json.loads(s) for s in row['text'].splitlines() if s]
  else:obj[name]=row['text'].strip()
 cases={};totals=dict(epochs=0,production=0,admission=0,formal=0,training_collection=0)
 for case in c['cases']:
  inp=read(R/'cases'/case/'INPUT.json');epochs=obj.get(case+'/EPOCHS_A.json',[]);outcomes=obj.get(case+'/OUTCOMES.json',[])
  expected=[(t,ep) for t in range(6,10) for ep in range(100)]
  assert [(x['stage'],x['epoch']) for x in epochs]==expected[:len(epochs)]
  steps=0
  for row in epochs:
   man=read(R/'cases'/case/'manifests'/('task%d.json'%row['stage']));n=(man['population']+9)//10;steps+=n
   assert row['batches']==n and row['steps']==row['physical_optimizer_calls']==steps
   target=man['bc_valid'] if inp['variant']=='ALIGN' else man['population']
   assert row['totals']['valid']==target and row['totals']['invalid']==man['population']-target
  ad=[]
  for stage in range(6,10):
   key=case+'/ADMISSION_A_%d.json'%stage
   if key in obj:
    x=obj[key];assert x['status']=='PASS_OWN_GPU_STRIPPED_S_ADMISSION' and x['optimizer_calls']==5 and x['environments']==0
    assert x['variant']==inp['variant'] and x['restore_initial_exact'] and x['roundtrip_exact'];ad.append(stage)
  assert not epochs or all(t in ad for t in set(x['stage'] for x in epochs))
  ids=[(x['stage'],x['task'],x['initial_index']) for x in outcomes]
  order=[(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)]
  assert ids==order[:len(ids)] and all(x['attempt']==1 and x['arm']=='A' for x in outcomes)
  if case+'/TRAIN_A.json' in obj:
   tr=obj[case+'/TRAIN_A.json'];assert tr['steps']==inp['production_budget']==steps and len(epochs)==400
   assert tr['physical_calls_observed']==dict(production=steps,admission=20) and tr['training_collection_episodes']==0
  if case+'/train_A.exit' in obj:assert obj[case+'/train_A.exit']=='0',('Training failed',case,obj[case+'/train_A.exit'])
  if case+'/evaluate.exit' in obj:assert obj[case+'/evaluate.exit']=='0',('Evaluation failed',case,obj[case+'/evaluate.exit'])
  cases[case]=dict(epochs=len(epochs),production=steps,admission=5*len(ad),admitted_stages=ad,formal=len(outcomes),variant=inp['variant'])
  for k in ['epochs','production','admission','formal']:totals[k]+=cases[case][k]
 assert 'coordinator/FAILURE.json' not in obj,obj.get('coordinator/FAILURE.json')
 if previous:
  old=read(Path(previous)/'EVIDENCE.json')['light']
  for name,row in old.items():
   if any(k in name for k in ['EPOCHS_A.json','OUTCOMES.json']) or name.endswith('.jsonl'):
    prior=json.loads(row['text']) if name.endswith('.json') else [json.loads(s) for s in row['text'].splitlines() if s]
    assert obj[name][:len(prior)]==prior,('Prefix changed',name)
   elif any(k in name for k in ['ADMISSION_','INITIAL_','STAGE_','SCHEDULE_','TRAIN_A.json','EVALUATION_COMPLETE','EVAL_ADMISSION','TASK6_FULL_S_COMPARISON']):
    assert light[name]['sha256']==row['sha256'],('Completed evidence changed',name)
 locks=obj.get('coordinator/ENVIRONMENT_LOCK_EVENTS.jsonl',[]);held=None;completed=0
 for row in locks:
  if row['kind']=='acquired':assert held is None;held=row['case']
  else:assert row['kind']=='released' and held==row['case'];held=None;completed+=1
 result=dict(status='PASS_PROGRESS_NOT_FINAL',observed=e['observed'],cases=cases,totals=totals,complete_environment_intervals=completed,environment_lock_held=held,prefix_preserved=bool(previous),contract=e['contract_sha256'],processes=e['processes'],controller_exit=obj.get('coordinator/controller.exit'))
 (d/'LOCAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result
if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)))
