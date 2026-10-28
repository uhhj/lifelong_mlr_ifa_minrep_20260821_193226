"""Independent existing-record/event audit only; no model or environment calls."""
import json,sys,hashlib,re,statistics
from pathlib import Path
D=Path(sys.argv[1]);seed=int(sys.argv[2]);js=lambda p:json.loads(p.read_text(encoding='utf8'));a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json')
assert a['failure'] is None
L=D/'light';records=js(L/'OUTCOMES.json');events=[json.loads(s) for s in (L/'EVAL_EVENTS.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
started={};completed={};cells={};bound=[]
for ev in events:
 run=ev['run'];m=re.fullmatch(r'([S])_s([6-9])_t([6-9])_g([01])',run);assert m
 arm,stage,task,group=m.groups();stage,task,group=map(int,(stage,task,group));assert task<=stage
 if ev['kind']=='start':
  assert run not in started;started[run]=ev
  assert js(L/f'STAGE_{arm}_{stage}.json')['sha256']==ev['cp_sha256']
 elif ev['kind']=='complete':
  assert run in started and run not in completed and ev['seconds']>0 and ev['time']>=started[run]['time'];completed[run]=ev
  rows=js(L/'outcomes'/(run+'.json'));assert len(rows)==10
  assert [r['initial_index'] for r in rows]==list(range(group*10,(group+1)*10))
  for r in rows:
   assert (r['arm'],r['stage'],r['task'],r['env_seed'])==(arm,stage,task,seed+group)
   assert r['cp_sha256']==started[run]['cp_sha256'] and r['initial_file_sha256']==started[run]['initial_file_sha256']
   assert r['termination']==('success' if r['success'] else 'timeout600') and r['attempt']==1
  bound.extend(rows);assert ev['episodes']==len(bound)
  cells.setdefault((arm,stage,task),set()).add(group)
 else:raise AssertionError(ev['kind'])
assert records[:len(bound)]==bound
# A snapshot may include a completed outcomes write before its complete event.
assert len(records)-len(bound) in (0,10)
ongoing=[x for k,x in started.items() if k not in completed];assert len(ongoing)<=1
for k in ['environments','updates']:assert js(L/'EVAL_ADMISSION.json')[k]==0
ad=js(L/'EVAL_ADMISSION.json');assert ad['status']=='PASS' and [r['length'] for r in ad['cache_parity']]==list(range(1,9))
times=[e['seconds'] for e in completed.values()];elapsed=max(0,v['observed']-ongoing[0]['time']) if ongoing else 0
out=dict(status='PASS_PROGRESS_NOT_FINAL',observation=v['observed'],episodes=len(records),completed_vectors=len(completed),closed_cells=sum(len(x)==2 for x in cells.values()),ongoing=ongoing,record_event_advance=len(records)-len(bound),vector_seconds=times,mean_seconds=statistics.mean(times) if times else None,current_elapsed=elapsed,remaining_mean_minutes=max(0,(20-len(completed))*statistics.mean(times)-elapsed)/60 if times else None,remaining_max_observed_minutes=max(0,(20-len(completed))*max(times)-elapsed)/60 if times else None,admission=ad,no_partial_performance_claim=True,limitations='Record/hash/attempt bindings and independently regenerated noise were checked locally; actual initial-file bytes and full CPU checkpoint audit remain final-only.',source_sha256={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ['OUTCOMES.json','EVAL_EVENTS.jsonl','EVAL_ADMISSION.json']})
(D/'EVALUATION_PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:x for k,x in out.items() if k not in ['admission','source_sha256']},indent=2))
