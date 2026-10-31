"""Recompute leakage boundaries from accepted evidence; no models or rollouts."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
B=js(R/'S_ACCEPTANCE.json');C=R.parent/'20261030_equal_optimizer_budget_cs_two_suites'
ca=js(C/'ACCEPTANCE.json');cf=Path(ca['final_directory']);cv=js(cf/'EVIDENCE.json');cfina=js(cf/'LOCAL_FINAL_AUDIT.json')
assert sha(cf/'LOCAL_FINAL_AUDIT.json')==ca['audit_sha256']
assert cfina['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and cfina['source_sha256']==sha(cf/'EVIDENCE.json')
def lights(v):
 for n,x in v['light'].items():assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],n
 return {n:json.loads(x['text']) for n,x in v['light'].items() if n.endswith('.json')}
cl=lights(cv);cases={};all_formal_states=set();sources={'C':dict(path=str(cf),evidence_sha256=sha(cf/'EVIDENCE.json'),audit_sha256=sha(cf/'LOCAL_FINAL_AUDIT.json'))}
for suite,ref in B['sources'].items():
 f=Path(ref['final_path']);v=js(f/'EVIDENCE.json');a=js(f/'LOCAL_FINAL_AUDIT.json')
 assert sha(f/'EVIDENCE.json')==ref['evidence_sha256']==a['evidence_sha256'] and sha(f/'LOCAL_FINAL_AUDIT.json')==ref['audit_sha256']
 assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
 sl=lights(v);sources[suite+'_S']=dict(path=str(f),evidence_sha256=sha(f/'EVIDENCE.json'),audit_sha256=sha(f/'LOCAL_FINAL_AUDIT.json'))
 for seed in [41,42,43]:
  case=f'{suite}_{seed}';sp=f'seed{seed}/';cp=case+'/';split=sl[sp+'INITIAL_SPLIT.json'];sr=sl[sp+'OUTCOMES.json'];cr=cl[cp+'OUTCOMES.json']
  assert len(sr)==len(cr)==200;stages={};collection=[]
  for task in range(6,10):
   z=split[str(task)];assert set(z['training'])==set(map(str,range(20,30))) and set(z['evaluation'])==set(map(str,range(20)))
   assert not set(z['training'].values())&set(z['evaluation'].values())
   collected=sl[sp+f'COLLECTION_OUTCOMES_S_{task}.json'];assert len(collected)==10
   assert {x['initial_index'] for x in collected}==set(range(20,30))
   m=sl[sp+f'MEMORY_S_{task}.json'];assert m['training_attempts']==10 and m['source_sha256']==sl[sp+f'STAGE_S_{task}.json']['sha256']
   for row in collected:
    assert row['attempt']==1 and row['initial_sha256']==z['training'][str(row['initial_index'])] and row['purpose']=='training_collection_not_formal_evaluation'
   for row in m['rows']:
    assert row['initial_index'] in range(20,30) and row['task']==task and row['source']=='online'
    assert collected[row['trajectory']]['success'] and collected[row['trajectory']]['initial_index']==row['initial_index']
   for arm,records in [('S',sr),('C',cr)]:
    for row in records:
     if row['task']==task:assert row['initial_index'] in range(20) and row['initial_sha256']==z['evaluation'][str(row['initial_index'])] and row['attempt']==1
   stages[str(task)]=dict(training_indices=list(range(20,30)),formal_indices=list(range(20)),exact_initial_hash_overlap=0,collection_episodes=10,cache_rows=m['count'],successful_cache_only=True,cache_cp_matches_stage=True)
   collection.extend(collected)
   if task>6:
    inp=sl[sp+f'TRAIN_INPUT_S_{task}.json'];assert set(inp['memory'])==set(map(str,range(6,task)))
    pr=sl[sp+f'PROBES_{task}.json'];assert pr['source']=='TRAINING_ONLY_FIXED_NOT_VALIDATION' and set(pr['support'])==set(map(str,range(6,task)))
    for t in range(6,task):
     old=sl[sp+f'MEMORY_S_{t}.json'];assert inp['memory'][str(t)]['cache_sha256']==old['cache_sha256']
     assert pr['support'][str(t)]==len({r['trajectory'] for r in old['rows']})/10
  sidx={(x['stage'],x['task'],x['initial_index']):x for x in sr}
  assert len(sidx)==200
  all_formal_states.update((suite,x['task'],x['initial_sha256']) for x in sr)
  for row in cr:
   s=sidx[row['stage'],row['task'],row['initial_index']]
   for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt']:assert row[k]==s[k]
  # Actual dispatch/event timing establishes no formal feedback during this run.
  formal_start=sl[sp+'EVALUATION_STARTED.json']['time']
  events=[json.loads(x) for x in v['light'][sp+'COLLECTION_EVENTS.jsonl']['text'].splitlines()]
  assert max(x['time'] for x in events)<formal_start
  assert cl[cp+'TRAIN_STATUS.json']['time']<cl[cp+'EVALUATION_STARTED.json']['time']
  cases[case]=dict(S_formal=200,C_formal=200,S_collection=40,stage_checks=stages,collection_formal_noise_hash_overlap=len({x['noise_sha256'] for x in collection}&{x['noise_sha256'] for x in sr}),formal_after_training=True,paired_C_S_inputs=True,unique_formal_initial_states=len({(x['task'],x['initial_sha256']) for x in sr}))
  assert cases[case]['collection_formal_noise_hash_overlap']==0
# Q is partial and is not treated as an accepted result.
rs=js(R/'RUN_STATUS.json');qf=R/rs['latest_evidence'];qv=js(qf/'EVIDENCE.json');ql=lights(qv);qchecks={}
for case in cases:
 count=0
 for task in range(6,10):
  n=case+f'/COLLECTION_OUTCOMES_Q_{task}.json'
  if n not in ql:continue
  z=ql[case+'/INITIAL_SPLIT.json'][str(task)];m=ql[case+f'/MEMORY_Q_{task}.json']
  assert not set(z['training'].values())&set(z['evaluation'].values())
  for row in ql[n]:assert row['initial_index'] in range(20,30) and row['initial_sha256']==z['training'][str(row['initial_index'])] and row['attempt']==1
  for row in m['rows']:assert row['initial_index'] in range(20,30) and row['source']=='online'
  count+=len(ql[n])
 qchecks[case]=dict(completed_collection_episodes=count,formal_episodes=len(ql.get(case+'/OUTCOMES.json',[])),not_accepted=True)
assert len(all_formal_states)==160
out=dict(status='PASS_SCOPED_RECORD_SEPARATION_NOT_GLOBAL_LEAKAGE_CERTIFICATE',observed=datetime.datetime.now().astimezone().isoformat(),scope='Latest six accepted S and six equal-budget C, plus latest recorded Q partial; historical baseline provenance shared but not all historical runs re-audited',sources=sources,cases=cases,Q=dict(snapshot=str(qf),sha256=sha(qf/'EVIDENCE.json'),counts=qchecks),accepted_formal_episodes=2400,S_training_collection_episodes=240,distinct_formal_task_initial_pairs_two_suites=len(all_formal_states),limits=['Same formal task/initial pairs reused across seeds/stages/method development, not 2400 independent held-out scenes','This test covers added online collection and recorded inputs, not pretraining data membership','Exact disjointness of added initial states is narrower than independent raw-demonstration distribution','S probes are training examples, not independent validation','Accepted execution correctness does not establish unbiased confirmatory generalization'])
(D/'RECORD_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],cases=len(cases),formal=2400,collection=240,Q=qchecks),ensure_ascii=False))
