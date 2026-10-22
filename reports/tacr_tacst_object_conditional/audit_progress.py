import json,sys,hashlib,math
from pathlib import Path
from formal_audit import audit_formal
p=Path(sys.argv[1]);d=json.loads((p/'EVIDENCE.json').read_text());assert not d['frozen_hashes']['failures'];result={}
for k,r in d['runs'].items():
 if not r['exists']:result[k]=dict(state='QUEUED_NOT_GPU_ADMITTED');continue
 assert not r['input_hashes']['failures']
 h=r['epochs'];led=r['ledger_by_epoch'];step=0;totals=dict(bc_valid=0,bc_invalid=0,effects=0,anchors=0,key_reads=0)
 for i,row in enumerate(h):
  assert (row['task'],row['epoch'])==(6+i//100,i%100)
  expected=math.ceil(r['entry']['spec']['dataset_sizes'][str(row['task'])]/10);step+=expected;assert row['steps']==step and row['batches']==expected
  a=led[i]
  if a.get('pending_atomic_write'):continue
  assert a['records']==expected and a['ids_complete'] and a['single_main_and_rep'] and a['steps']==list(range(step-expected,step)) and a['residual_ratio_max']<=.200001
  for n in totals:totals[n]+=a[n]
 pre=r['status_files']['GPU_PREFLIGHT.json']
 if pre:assert pre['status']=='PASS' and pre['optimizer_calls']==8 and pre['main_optimizer_calls']==5 and pre['representation_optimizer_calls']==3 and pre['environment_steps']==0
 for name,rc in [*r['phase_exits'].items(),*r['admission'].items()]:
  if rc is not None:assert rc.strip()=='0',(k,name,rc)
 formal=audit_formal(r['formal'],r['entry']['seed'])
 result[k]=dict(epochs=len(h),main_steps=step,representation_steps=step,formal=formal,exposures=totals,preflight_pass=bool(pre))
state=d['queue_state']['RUN_STATUS.json'];assert not d['queue_state']['FAILURE.json']
if state['status']=='WAITING_PRIOR_QUEUE_NATURAL_EXIT':assert all(not r['exists'] for r in d['runs'].values())
a=dict(status='PASS_PROGRESS_NOT_FINAL',observed_unix=d['observed_unix'],frozen_files=d['frozen_hashes']['count'],queue_status=state['status'],runs=result,evidence_sha256=hashlib.sha256((p/'EVIDENCE.json').read_bytes()).hexdigest())
(p/'LOCAL_AUDIT.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
