"""Local complete-training/recovery metadata audit; no model or environment calls."""
import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;S=Path(sys.argv[1])
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=js(S/'EVIDENCE.json');a=js(S/'LOCAL_AUDIT.json');ev=js(S/'EVALUATION_PROGRESS_AUDIT.json');B=S/'metadata/B2'
assert not d['hash_bad'] and d['frozen_files']==1875
assert d['status']['phase']=='B2_evaluate' and d['status']['controller']==2086
assert {2086,d['status']['pid']}<=set(x['pid'] for x in d['own_processes'])
assert str(d['status']['pid']) in d['gpu'] and d['controller_exit'] is None
assert a['arms']['B2']==dict(complete_epochs=400,main_updates=105500)
complete=js(B/'TRAINING_COMPLETE.json');cost={};binding={}
for t in range(6,10):
 for phase in ('train','diagnose'):assert d['phase_exits'][f'B2_{phase}{t}.exit']=='0'
 rows=js(B/'epochs'/f'task{t}.json');base=js(S/'metadata/B1/epochs'/f'task{t}.json')
 ds=js(B/'dataset'/f'task{t}.json')['rows']
 assert ds==js(S/'metadata/B1/dataset'/f'task{t}.json')['rows']==js(S/'metadata/B0/dataset'/f'task{t}.json')['rows']
 assert [x['order_sha256'] for x in rows]==[x['order_sha256'] for x in base]
 assert js(B/'optimizer'/f'task{t}.json')==js(S/'metadata/B1/optimizer'/f'task{t}.json')
 if t==6:
  assert [{k:v for k,v in x.items() if k not in ('seconds','arm')} for x in rows]==[{k:v for k,v in x.items() if k not in ('seconds','arm')} for x in base]
  assert all(x['transfer_steps']==x['retained_steps']==0 for x in rows)
 else:
  teacher=js(B/'teachers'/f'task{t}.json');assert teacher==js(B/'TRANSFER_ADMISSION_BY_STAGE'/f'task{t}.json')
  assert teacher['binding']==dict(contract=d['contract_sha256'],codec_sha256=complete['codec_sha256'],prior_policy_sha256=complete['checkpoints'][str(t-1)],dataset_sha256=sha(B/'dataset'/f'task{t}.json'))
  old_valid=sum(sum(x['action_mask']) for x in ds if x['task']<t)
  assert all(x['transfer_anchors']==teacher['accepted'] and x['retained_steps']==old_valid and 0<=x['transfer_steps']<=4*teacher['accepted'] for x in rows)
  binding[str(t)]=teacher
 cost[str(t)]={k:sum(x[k] for x in rows) for k in ('seconds','action_valid','action_invalid','effect_valid','transfer_anchors','transfer_steps','retained_steps','extra_retention_decoder_windows')}
old=js(R/'snapshot_20261024_132137/metadata/B2/epochs/task8.json')
assert len(old)==90 and js(B/'epochs/task8.json')[:90]==old
resolution=js(S/'metadata/shared/RECOVERY06_RESOLUTION.json')
assert resolution['preserved_prefix_epochs']==90 and resolution['repeated_completed_epochs']==0 and resolution['task8_final_steps']==69300
assert resolution['recovery_execution_contract']=='34d1fe1b29502d55f183c8ccd7d9a6ba8266768421374c2561f724904d22a61b'
assert resolution['admission_sha256']==sha(S/'metadata/shared/RECOVERY06_ADMISSION.json')
durations=ev['vector_seconds'];avg=sum(durations)/len(durations)
remaining=max(0,((20-len(durations))*avg-sum(x['elapsed_seconds_at_snapshot'] for x in ev['ongoing_vectors']))/60)
out=dict(status='PASS_COMPLETE_B2_TRAINING_METADATA_NOT_FINAL',observation_unix=d['unix'],phase=d['status'],source_contract=d['contract_sha256'],complete_epochs=400,main_updates=105500,dataset_identity_order_mask_equal=True,sampler_order_equal_B1=True,optimizer_whitelist_equal_B1=True,Task6_all_non_timing_epoch_fields_equal_B1=True,preserved_Task8_prefix90_exact=True,recovery_resolution=resolution,teacher_bindings=binding,epoch_accounting=cost,epoch_minutes=sum(x['seconds'] for x in cost.values())/60,interrupted_uncommitted_epoch_cost='UNKNOWN_SEPARATE',complete_formal_episodes=ev['complete_episodes'],complete_vectors=len(durations),vector_mean_seconds=avg,remaining_linear_minutes_at_snapshot=remaining,next_check_minutes=90,estimate_limit='Only first two B2 vectors, future throughput uncertain; stage loading/server audit/local final acceptance extra.',full_independent_acceptance='PENDING')
(S/'B2_TRAINING_AND_TRANSITION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('teacher_bindings','epoch_accounting','recovery_resolution')},indent=2))
