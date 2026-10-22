import json,math,hashlib,sys,datetime
from pathlib import Path
from formal_audit import audit_formal
p=Path(sys.argv[1]);d=json.loads((p/'EVIDENCE.json').read_text());assert not d['frozen_hashes']['failures']
assert d['queue_state']['FAILURE.json'] is None
runs={}
for key,r in d['runs'].items():
 assert r['worker']['FAILURE.json'] is None
 assert all(v is None or str(v).strip()=='0' for v in r['admission'].values())
 assert r['worker']['supervisor.exit'] is None or r['worker']['supervisor.exit'].strip()=='0'
 if not r['exists']:
  runs[key]=dict(status='QUEUED_NOT_GPU_ADMITTED',formal_records=0);continue
 assert not r['input_hashes']['failures'] and r['initial_matches_parent'] is not False
 h=r['epochs'];step=0
 for i,x in enumerate(h):
  t=6+i//100;ep=i%100;n=r['entry']['spec']['dataset_sizes'][str(t)]
  assert (x['task'],x['epoch'],x['batches'])==(t,ep,math.ceil(n/10));step+=x['batches'];assert x['steps']==step
 for x in r['ledger_by_epoch']:
  if x.get('pending_atomic_write'):continue
  hh=next(y for y in h if (y['task'],y['epoch'])==(x['task'],x['epoch']));assert x['records']==hh['batches'] and x['ids_complete'] and x['native_only']
  assert x['steps']==list(range(hh['steps']-hh['batches'],hh['steps']))
 gate=r['status_files']['GPU_PREFLIGHT.json']
 if h:
  assert gate['status']=='PASS' and gate['optimizer_calls']==gate['main_optimizer_calls']==5 and gate['auxiliary_optimizer_calls']==gate['environment_steps']==0
 formal=audit_formal(r['formal'],r['entry']['seed'])
 last=h[-1] if h else None;recent=[x['seconds'] for x in h[-10:] if x['task']==last['task']] if h else []
 runs[key]=dict(status='SERVER_ACCEPTED_PENDING_LOCAL_FINAL' if r['receipt'] else 'RUNNING_OR_PREPARING',complete_epochs=len(h),main_steps=step,last=last,formal=formal,gpu_admission=gate['status'] if gate else 'NOT_YET',recent_stage_epoch_seconds=sum(recent)/len(recent) if recent else None)
out=dict(status='PASS',observed_at=datetime.datetime.fromtimestamp(d['observed_unix']).astimezone().isoformat(),evidence_sha256=hashlib.sha256((p/'EVIDENCE.json').read_bytes()).hexdigest(),frozen_hash_count=d['frozen_hashes']['count'],runs=runs,formal_acceptance=False)
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
