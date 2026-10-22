"""Local recomputation of observed completed epoch accounting."""
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);raw=(p/'SNAPSHOT.json').read_bytes();d=json.loads(raw)
assert not d['hashes']['failures'];runs={}
for name,r in d['runs'].items():
 h=r['epochs'];gate=r['files']['GPU_PREFLIGHT.json'];failure=r['worker']['FAILURE.json']
 if failure:raise AssertionError((name,failure))
 steps=0
 for row in h:
  expected={6:157,7:227,8:309,9:362}[row['task']];steps+=expected
  assert row['batches']==expected and row['steps']==steps
 assert [(x['task'],x['epoch']) for x in h]==[(t,e) for t in (6,7,8,9) for e in range(100)][:len(h)]
 if r.get('inputs'):assert not r['inputs']['failures']
 if r['files']['ra_coreset/INITIAL_STATE.json'] is not None:assert r['files']['ra_coreset/INITIAL_STATE.json']==r.get('parent_initial',r['files']['ra_coreset/INITIAL_STATE.json'])
 if h:assert gate and gate['status']=='PASS' and gate['environment_steps']==0 and gate['proof']['disabled_full_state_exact_RA'] and gate['proof']['active_serialized_full_state_restore_exact']
 assert sum(x['records'] for x in r['auxiliary'].values())>=steps
 assert all(str(v).strip()=='0' for k,v in r['exits'].items())
 recent=[x['seconds'] for x in h[-10:] if x['task']==h[-1]['task']] if h else []
 runs[name]=dict(complete_epochs=len(h),steps=steps,formal_records=r['formal_records'],gpu_admission=gate['status'] if gate else 'PENDING',last=h[-1] if h else None,last_stage_mean_epoch_seconds=sum(recent)/len(recent) if recent else None,mechanism=r['auxiliary'])
out=dict(status='PASS',snapshot_sha256=hashlib.sha256(raw).hexdigest(),frozen_hashes=d['hashes']['count'],runs=runs,formal_acceptance=False)
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
