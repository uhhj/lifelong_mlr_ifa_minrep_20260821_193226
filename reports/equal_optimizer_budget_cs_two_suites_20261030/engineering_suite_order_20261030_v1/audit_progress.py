"""Independent metadata and paired-record audit; no model or environment calls."""
import sys,json,hashlib,statistics,re
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));v=js(D/'EVIDENCE.json');L={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];p=D/'light'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x['text'],encoding='utf8');L[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
cases=['GOAL_41', 'GOAL_42', 'GOAL_43', 'OBJECT_41', 'OBJECT_42', 'OBJECT_43'];counts={};rates={};admissions={};teachers={};evaluations={};failure=L.get('FAILURE.json') or L.get('engineering_suite_order_20261030_v1/FAILURE.json')
for case in cases:
 parent=js(R/case/'PARENT_BINDING.json');seed=parent['seed'];stages=parent['stages'];prefix=case+'/'
 def get(n,default=None):return L.get(prefix+n,default)
 rows=get('EPOCHS_C.json',[]);steps=0
 for i,row in enumerate(rows):
  t=6 if i<100 else 7+(i-100)//200;ep=i if i<100 else (i-100)%200;st=stages[str(t)];steps+=st['batches']
  assert (row['stage'],row['epoch'],row['batches'],row['steps'])==(t,ep,st['batches'],steps)
  assert all(row['totals'][k]==val for k,val in [('valid',st['valid']),('invalid',st['population']-st['valid']),('old_eligible',st['old_valid'])])
  assert row['physical_optimizer_calls']==steps
  assert row['isolated_optimizer_calls']==4+29*(t-6)
  schedule=get('SCHEDULE_C_%d.json'%t);assert schedule['total_steps']==st['batches']*(100 if t==6 else 200) and schedule['warmup_steps']==500
  key=str(sum(stages[str(j)]['batches']*(100 if j==6 else 200) for j in range(6,t)))
  assert get('COPY_EQ.json')['checked'][key]['all_state_exact']
 counts[case]=dict(epochs=len(rows),steps=steps,by_stage={str(t):sum(r['stage']==t for r in rows) for t in range(6,10)})
 rates[case]={str(t):statistics.mean([r['seconds'] for r in rows if r['stage']==t][-10:]) for t in range(6,10) if any(r['stage']==t for r in rows)}
 for t in range(6,10):
  a=get('ADMISSION_C_%d.json'%t)
  if a:
   assert a['status']=='PASS' and a['roundtrip_exact'] and a['next_step_exact'] and a['restore_initial_exact'] and a['strict_keys']==100 and a['parameters']==21633611 and a['environments']==0 and a['baseline_or_legacy_update_exact']
   assert a['optimizer_calls']==(4 if t==6 else 29)
   if t>6:assert a['stateful_gradient_measurement_exact'] and a['ten_step_and_checkpoint_continuation_exact']
   admissions[case+'_'+str(t)]=a['optimizer_calls']
  m=get('TEACHER_C_%d.json'%t)
  if m:
   st=stages[str(t)];expected={i:r for i,r in enumerate(st['rows']) if r['task']<t and r['bc_valid']}
   assert m['count']==len(expected)==st['old_valid'] and m['manifest_sha256']==st['manifest_sha256'] and {r['index'] for r in m['rows']}==set(expected)
   assert m['source_sha256']==get('STAGE_C_%d.json'%(t-1))['sha256']
   for r in m['rows']:
    assert r['sample_sha256']==expected[r['index']]['sha256'] and r['task']==expected[r['index']]['task']
    # Frozen teacher weights use FP32 CUDA arithmetic, not Python float64.
    ref=np.float32(m['tau'])/(np.float32(m['tau'])+np.float32(r['teacher_cost']))
    assert np.isfinite(r['weight']) and 0<=r['weight']<=1
    assert abs(r['weight']-float(ref))<=float(np.spacing(ref))
   assert abs(m['tau']-max(1e-8,statistics.median(r['teacher_cost'] for r in m['rows'])))<1e-14
   teachers[case+'_'+str(t)]=dict(count=m['count'],tau=m['tau'],cache_sha256=m['cache_sha256'],source_sha256=m['source_sha256'])
  sm=get('STAGE_C_%d.json'%t)
  if sm:assert sm['epochs']==(100 if t==6 else 200) and sm['steps']==sum(stages[str(j)]['batches']*(100 if j==6 else 200) for j in range(6,t+1))
 if get('train_C.exit') is not None:
  if get('train_C.exit')!='0':failure=dict(case=case,exit=get('train_C.exit'))
  else:assert len(rows)==700 and steps==sum(st['batches']*(100 if int(t)==6 else 200) for t,st in stages.items())
 if get('INITIAL_C.json'):
  initial=get('INITIAL_C.json');assert initial['status']=='PASS_OFFICIAL_INITIALIZATION' and initial['official_cp_sha256']==parent['official_initialization']['sha256']
 recs=get('OUTCOMES.json',[]);seen=set();bind={}
 for r in recs:
  key=(r['arm'],r['stage'],r['task'],r['initial_index']);assert key not in seen;seen.add(key)
  assert r['arm']=='C' and 6<=r['task']<=r['stage']<=9 and r['initial_index'] in range(20) and r['attempt']==1 and r['env_seed']==seed+r['initial_index']//10
  assert 0<r['steps']<=600 and (r['success'] or r['steps']==600) and r['termination']==('success' if r['success'] else 'timeout600')
  split=parent['initial_split'][str(r['task'])];assert r['initial_file_sha256']==split['file_sha256'] and r['initial_sha256']==split['evaluation'][str(r['initial_index'])]
  noise=np.random.RandomState(20261024+seed*1000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32)
  assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256'] and r['cp_sha256']==get('STAGE_C_%d.json'%r['stage'])['sha256']
  ik=(r['task'],r['initial_index']);binding=(r['initial_sha256'],r['noise_sha256']);assert bind.setdefault(ik,binding)==binding
 if get('EVAL_ADMISSION.json'):
  a=get('EVAL_ADMISSION.json');assert a['status']=='PASS' and a['environments']==a['updates']==0 and [x['length'] for x in a['cache_parity']]==list(range(1,9))
 events=[json.loads(s) for s in get('EVAL_EVENTS.jsonl','').splitlines() if s.strip()];started={};completed={};bound=[];cells={}
 for e in events:
  run=e['run'];m=re.fullmatch(r'C_s([6-9])_t([6-9])_g([01])',run);assert m;t,k,g=map(int,m.groups());assert k<=t
  if e['kind']=='start':
   assert run not in started;started[run]=e;assert e['cp_sha256']==get('STAGE_C_%d.json'%t)['sha256']
  else:
   assert e['kind']=='complete' and run in started and run not in completed and e['seconds']>0
   rr=get('outcomes/'+run+'.json');assert len(rr)==10 and [r['initial_index'] for r in rr]==list(range(g*10,(g+1)*10))
   assert all((r['arm'],r['stage'],r['task'])==('C',t,k) and r['cp_sha256']==started[run]['cp_sha256'] and r['initial_file_sha256']==started[run]['initial_file_sha256'] for r in rr)
   bound+=rr;assert e['episodes']==len(bound);completed[run]=e;cells.setdefault((t,k),set()).add(g)
 assert recs[:len(bound)]==bound and len(recs)-len(bound) in (0,10)
 ongoing=[x for k,x in started.items() if k not in completed];assert len(ongoing)<=1
 times=[e['seconds'] for e in completed.values()]
 evaluations[case]=dict(episodes=len(recs),vectors=len(completed),cells=sum(len(x)==2 for x in cells.values()),ongoing=ongoing,vector_seconds=times,record_event_advance=len(recs)-len(bound))
 if get('evaluate.exit') is not None:
  if get('evaluate.exit')!='0':failure=dict(case=case,evaluate_exit=get('evaluate.exit'))
  else:assert len(recs)==200 and len(completed)==20 and len(cells)==10 and not ongoing
if not failure and L.get('engineering_suite_order_20261030_v1/controller.exit')!='0' and not v.get('processes'):failure={'reason':'No successful controller exit and no own process'}
prefix_verified=None
if len(sys.argv)>2:
 previous=js(Path(sys.argv[2])/'EVIDENCE.json')
 assert previous['contract_sha256']==v['contract_sha256']
 for n,x in previous['light'].items():
  if n.endswith(('EPOCHS_C.json','OUTCOMES.json','GRADIENTS_C.json')):
   old=json.loads(x['text']);assert L[n][:len(old)]==old,('prefix',n)
  elif n.endswith('EVAL_EVENTS.jsonl'):assert L[n].startswith(x['text']),('prefix',n)
 prefix_verified=str(sys.argv[2])

# Independently bind the new scheduling authority; do not fabricate root success.
assert 'controller.exit' not in L, 'Original coordinator was superseded, not successful'
assert v['engineering_contract']==js(R/'engineering_suite_order_20261030_v1/deploy_remote_20261030_162036/EVIDENCE.json')['engineering_contract']
assert v['engineering_files']==3
eng='engineering_suite_order_20261030_v1/'
if eng+'HANDOFF_COMPLETE.json' in L:
 receipt=L[eng+'GOAL42_EXIT_OBSERVATION.json']
 assert receipt['status']=='NATURAL_GOAL42_EXIT_VERIFIED' and receipt['kernel_exit']['state']=='Z' and receipt['kernel_exit']['exit_code']==0 and receipt['training_signals']==0
 assert L['GOAL_42/train_C.exit']=='0'
if eng+'QUEUE.json' in L:
 expected=[['train','GOAL_43']]+[['evaluate','GOAL_'+str(s)] for s in [41,42,43]]+[['train','OBJECT_'+str(s)] for s in [41,42,43]]+[['evaluate','OBJECT_'+str(s)] for s in [41,42,43]]
 assert L[eng+'QUEUE.json']['jobs']==expected
if any(counts['OBJECT_'+str(s)]['epochs'] for s in [41,42,43]):
 assert all(evaluations['GOAL_'+str(s)]['episodes']==200 and L['GOAL_'+str(s)+'/evaluate.exit']=='0' for s in [41,42,43])

out=dict(engineering_contract=v['engineering_contract'],engineering_files=v['engineering_files'],completion_source='engineering_suite_order_20261030_v1/controller.exit',prefix_verified=prefix_verified,status='FAILURE_REQUIRES_INSPECTION' if failure else 'PASS_PROGRESS_NOT_FINAL',observation=v['observed'],contract=v['contract_sha256'],source_files=v['source_files'],counts=counts,rates_last10=rates,admissions=admissions,isolated_optimizer_calls=sum(admissions.values()),teachers=teachers,evaluations=evaluations,formal_episodes=sum(e['episodes'] for e in evaluations.values()),phase=L.get('engineering_suite_order_20261030_v1/STATUS.json',L.get('STATUS.json')),failure=failure,processes=v.get('processes',[]),source_evidence_sha256=sha(D/'EVIDENCE.json'),snapshot_sha256=sha(D/'EVIDENCE.json'),no_partial_performance_claim=True)
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:x for k,x in out.items() if k not in ['teachers','evaluations','processes']},indent=2))
