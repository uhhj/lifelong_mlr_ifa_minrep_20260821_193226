"""Audit resumed progress and account for recovery diagnostics separately."""
import hashlib,json,sys
from pathlib import Path
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'))
t=json.loads((D/'TRANSITION_AUDIT.json').read_text(encoding='utf8'))
E='engineering_resume_20261028_v1/'
def js(n):return json.loads(v['light'][n]['text'])
assert a['status']==t['status']=='PASS_PROGRESS_NOT_FINAL'
assert len(v['frozen'])==56
assert v['engineering']['contract_sha256']=='f853cc7498f0a634e8d8a5c631630bfd0353452d525bfda5f14c0721bdb77050'
contract=js(E+'CONTRACT.json')
assert v['engineering']['files']==contract['files']
r=js(E+'RECOVERY_ADMISSION.json');g=js(E+'GPU_ADMISSION.json')
assert r['status']=='PASS_GPU_FULL_STATE_RESTORE_AND_NEXT_STEP_REPLAY'
assert r['checkpoint_sha256']=='df2c9101552a50f0478b36b1df942166c614b598a6d11af0d595fb4c9da7bb97'
assert r['optimizer_calls']==31 and r['extra_autograd_calls']==11 and r['environments']==0
assert all(r[k] for k in ['full_state_exact','next_step_replay_exact','checkpoint_restored_after_admission'])
assert g['status']=='PASS_OWN_GPU_ADMISSION' and g['optimizer_calls']==29
assert g['accepted_and_rejected_full_state_restore'] and g['probe_all_state_exact']
started=js(E+'STARTED.json');assert len(started['completed_epoch_file_sha256'])==13
for key,h in started['completed_epoch_file_sha256'].items():
    case,arm=key.split('/');assert v['light'][case+'/EPOCHS_'+arm+'.json']['sha256']==h
history=js('OBJECT_41/EPOCHS_S.json');original=js(E+'original_OBJECT_41_EPOCHS_S.json')
assert len(original)==7 and history[:7]==original
assert len(history)>7 and 'controller.exit' not in v['light']
resumed=history[7:];mean=sum(x['seconds'] for x in resumed[-10:])/len(resumed[-10:])
out=dict(status='PASS_RESUME_PROGRESS_NOT_FINAL',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],engineering_contract_sha256=v['engineering']['contract_sha256'],original_56_frozen_exact=True,completed_13_epoch_files_exact=True,original_S_7_epochs_exact=True,resumed_complete_epochs=len(resumed),S_complete_epochs=len(history),recovery_admission=r,original_own_isolated_optimizer_calls=a['own_isolated_optimizer_calls'],extra_recovery_optimizer_calls=31,total_known_isolated_optimizer_calls=a['own_isolated_optimizer_calls']+31,extra_recovery_autograd_calls=11,interrupted_partial_epoch_optimizer_calls='UNKNOWN',accepted_trajectory_physical_optimizer_calls=a['physical_training_optimizer_calls'],resumed_recent_epoch_mean_seconds=mean,S_remaining_compute_minutes_estimate=(100-len(history))*mean/60,estimate_excludes_saving_preparation_and_QGU_evaluation=True,final_CPU_and_closed_loop_pending=True)
(D/'RESUME_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
