"""Read-only partial count/storage/attempt audit; never infer missing metrics."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys,datetime,statistics
R=Path(__file__).resolve().parent;D=R/sys.argv[1];E=D/'baseline_evidence'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
A=js(D/'FROZEN_INPUT_AUDIT.json');S=js(D/'SNAPSHOT.json')
for n,h in A['manifest'].items():assert sha(E/n)==h['sha256'] and (E/n).stat().st_size==h['bytes']
assert A['source_config_input_modal_features_orchestration_verified']
assert js(E/'TASK6_TENSOR_PARITY.json')['equal']
assert js(E/'random/INITIAL_STATE.json')==js(E/'coreset/INITIAL_STATE.json')
features=js(E/'FEATURES.json')['hashes'];out={};dur=[]
for arm in ['random','coreset']:
 ep=js(E/arm/'epochs.json');assert len(ep)==400 and {(e['task'],e['epoch']) for e in ep}=={(t,k) for t in range(6,10) for k in range(100)}
 assert sum(e['batches'] for e in ep)==76100 and ep[-1]['steps']==76100
 complete=js(E/arm/'TRAINING_COMPLETE.json')
 for n,h in complete['checkpoints'].items():assert A['checkpoints'][arm+'/models/'+n]['sha256']==h
 buffer={}
 for t,k in zip(range(6,10),[499,445,452,985]):
  m=js(E/arm/'storage'/f'task{t}.json');assert m['K']==k and len(m['selected'])==k
  assert len({v['destination'] for v in m['selected']})==k
  for v in m['selected']:
   assert features[str(t)+'/'+Path(v['source']).name]==v['sha256']
   assert v['destination'] not in buffer;buffer[v['destination']]=v['sha256']
 assert js(E/arm/'buffer_after_task9.json')['hashes']==buffer
 rows=[json.loads(l) for p in (E/arm/'evaluation').glob('trials*.jsonl') for l in p.read_text().splitlines()]
 assert len({(r['stage'],r['task'],r['trial']) for r in rows})==len(rows)
 for r in rows:
  assert r['seed']==41 and r['arm']==arm and r['epoch']==99 and 6<=r['task']<=r['stage']<=9 and 0<=r['trial']<20 and type(r['success']) is bool
  assert r['checkpoint_sha256']==complete['checkpoints'][f"mlrifa_{r['stage']}_model_ep99.pth"]
 events=[json.loads(l) for p in (E/arm/'evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines()]
 counts=Counter(e['event'] for e in events);assert counts['technical_exception']==0
 starts={(e['stage'],e['task'],e['group']):e for e in events if e['event']=='vector_attempt_start'}
 done={(e['stage'],e['task'],e['group']):e for e in events if e['event']=='vector_attempt_complete'}
 assert len(rows)==10*len(done)
 for key,e in done.items():
  assert key in starts
  rr=sorted([r for r in rows if (r['stage'],r['task'],r['trial']//10)==key],key=lambda x:x['trial'])
  assert [r['success'] for r in rr]==e['successes']
  assert e['env_seed']==41+key[2]
  # Ledger wall-clock field is discovered explicitly; no fabricated timing.
  tk='time' if 'time' in e else 'unix' if 'unix' in e else 'timestamp' if 'timestamp' in e else None
  if tk and isinstance(e[tk],(int,float)):dur.append(e[tk]-starts[key][tk])
 cells=[]
 for key in sorted({(r['stage'],r['task']) for r in rows}):
  rr=[r for r in rows if (r['stage'],r['task'])==key]
  cells.append(dict(stage=key[0],task=key[1],trials=len(rr),successes=sum(r['success'] for r in rr),complete_cell=len(rr)==20))
 out[arm]=dict(epochs=400,steps=76100,checkpoints_present=4,strict_final_checkpoint_acceptance='PENDING',records=len(rows),vector_attempts_complete=len(done),vector_attempts_started=len(starts),technical_exceptions=0,storage_quota_and_manifest_pass=True,training_minutes=complete['seconds_this_invocation']/60,partial_cells=cells)
remain=60-sum(v['vector_attempts_complete'] for v in out.values())
estimate_vector_seconds=statistics.mean(dur) if dur else None
estimate_hours=(remain*estimate_vector_seconds+41*60+15*60)/3600 if dur else None
result=dict(status='PASS_PARTIAL_NOT_FINAL_ACCEPTANCE',observed_unix=A['time'],source_input_hashes_pass=True,task6_actual_tensor_parity=True,initial_rng_parity=True,arms=out,ra_status='QUEUED_AFTER_BASELINE_ACCEPTANCE',total_epochs=800,total_steps=152200,total_records=sum(v['records'] for v in out.values()),expected_records=600,full_metrics=None,completed_vector_seconds=dur,mean_vector_seconds=estimate_vector_seconds,remaining_hours_point_estimate=estimate_hours,remaining_RA_training_minutes_estimate=41,overhead_minutes_assumption=15,new_training_or_environment_calls=0)
(D/'PROGRESS_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
