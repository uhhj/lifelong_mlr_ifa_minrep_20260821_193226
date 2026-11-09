"""Independent local budget, sequence, lock and append-only evidence audit."""
from pathlib import Path
import json,sys,hashlib,math
R=Path(__file__).resolve().parent;D=Path(sys.argv[1])
def js(p):return json.loads(p.read_text(encoding='utf8'))
e=js(D/'EVIDENCE.json');L={}
for n,v in e['light'].items():
 assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256']
 L[n]=v['text']
counts={}
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  case=suite+'_'+str(seed);prefix=case+'/';rows=json.loads(L.get(prefix+'random/epochs.json','[]'));previous_steps=0
  for idx,r in enumerate(rows):
   t=6+idx//100;epoch=idx%100;m=js(R/'cases'/case/'manifests'/('task%d.json'%t));b=m['batches'];assert r['task']==t and r['epoch']==epoch and r['batches']==b
   previous_steps+=b;assert r['steps']==previous_steps and math.isfinite(r['loss']) and r['seconds']>0
   def lr(step):return 1e-4*(step/500 if step<500 else max(0.,(100*b-step)/(100*b-500)))
   assert math.isclose(r['lr_first'],lr(epoch*b),rel_tol=1e-10,abs_tol=1e-14)
   assert math.isclose(r['lr_after'],lr((epoch+1)*b),rel_tol=1e-10,abs_tol=1e-14)
  trials=[]
  for n,text in L.items():
   if n.startswith(prefix+'random/evaluation/trials_'):trials.extend(json.loads(x) for x in text.splitlines() if x)
  keys=[(r['stage'],r['task'],r['initial_state_index']) for r in trials];assert len(keys)==len(set(keys))
  assert all(r['seed']==seed and 0<=r['initial_state_index']<20 and 6<=r['task']<=r['stage']<10 for r in trials)
  admissions={}
  for t in range(6,10):
   name=prefix+'ADMISSION_%d.json'%t
   if name in L:
    a=json.loads(L[name]);assert a['status']=='PASS_OWN_GPU_ALIGN_ADMISSION' and a['optimizer_calls']==5 and a['production_calls_added']==a['environments']==0;admissions[str(t)]=a
  counts[case]=dict(epochs=len(rows),production_updates=previous_steps,formal_episodes=len(trials),admissions=len(admissions),admission_updates=5*len(admissions))
events=[json.loads(x) for x in L.get('coordinator/EVENTS.jsonl','').splitlines() if x];held=None;locks=0
for x in events:
 if x['kind']=='environment_acquired':assert held is None;held=(x['case'],x['pid'])
 if x['kind']=='environment_released':assert held==(x['case'],x['pid']);held=None;locks+=1
if len(sys.argv)>2:
 old=js(Path(sys.argv[2])/'EVIDENCE.json')['light']
 for n,v in old.items():
  if n.endswith('/random/epochs.json'):
   a=json.loads(v['text']);b=json.loads(L[n]);assert b[:len(a)]==a
  elif n.endswith('.jsonl'):assert L[n].startswith(v['text']),n
  elif '/ADMISSION_' in n or n.endswith('/INITIAL_STATE.json'):assert L[n]==v['text'],n
out=dict(status='PASS_PROGRESS_NOT_FINAL',observed=e['observed'],counts=counts,epochs=sum(x['epochs'] for x in counts.values()),production_updates=sum(x['production_updates'] for x in counts.values()),formal_episodes=sum(x['formal_episodes'] for x in counts.values()),environment_mutex=dict(completed=locks,active=held),new_training_or_rollouts=0)
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
