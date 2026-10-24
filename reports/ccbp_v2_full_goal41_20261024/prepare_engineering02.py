"""Versioned device-comparison repair and exact completed-Task6 continuation."""
from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;backup=R/'engineering_attempt_01';backup.mkdir(exist_ok=False)
bundle=json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'))
for n,s in bundle.items():(backup/n).write_text(s,encoding='utf8')
old=bundle['train.py'];new=old.replace('assert equal(a.policy.state_dict(),previous)','assert equal(cpu(a.policy.state_dict()),previous)')
new=new.replace("dest=OUTPUT[arm];dest.mkdir(parents=True,exist_ok=True);assert not (dest/'recovery.pth').exists(),'No blind resume'","dest=OUTPUT[arm];dest.mkdir(parents=True,exist_ok=True);resume=(arm=='B' and (R/'REPAIR_CONTINUATION.json').exists());assert resume or not (dest/'recovery.pth').exists(),'No blind resume'")
new=new.replace("a=make(arm);initialization(a,arm);history=[];gradients=[]\n  for stage in range(6,10):",'''a=make(arm);initialization(a,arm);history=[];gradients=[];first=6
  if resume:
   binding=js(R/'REPAIR_CONTINUATION.json');rec=torch.load(dest/'recovery.pth',map_location='cpu')
   assert sha(dest/'recovery.pth')==binding['recovery_sha256'] and rec['contract']==binding['previous_contract'] and rec['stage']==6 and rec['next_epoch']==100 and rec['steps']==10000
   setup(a,Dataset(6));restore(a,rec['state']);assert equal(capture(a),rec['state'])
   history=rec['history'];assert history==js(R/'EPOCHS_B.json') and len(history)==100
   write(R/'RESTORE_ADMISSION.json',dict(status='PASS_ACTUAL_GPU_FULL_STATE_RESTORE',recovery_sha256=binding['recovery_sha256'],previous_contract=binding['previous_contract'],next_stage=7,completed_epochs_repeated=0,optimizer_updates=0,forwards=0,environments=0,fields=sorted(rec['state']),all_fields_exact=True))
   first=7
  for stage in range(first,10):''')
assert new!=old;ast.parse(new);bundle['train.py']=new;(R/'train.py').write_text(new,encoding='utf8')
(R/'DEPLOY_TEXT_BUNDLE_ENGINEERING02.json').write_text(json.dumps(bundle,indent=2),encoding='utf8')
# External observation/final scripts follow the active version; original server source is unchanged.
for n in ['check_readonly.py','final_readonly.py','build_final_wrapper.py']:
 p=R/n;s=p.read_text(encoding='utf8');s=s.replace('/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1','/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02');p.write_text(s,encoding='utf8')
report=dict(status='ENGINEERING_REPAIR_PREPARED_NOT_DEPLOYED',cause='torch.equal compared CUDA student state with CPU-loaded previous CP',scientific_change=False,changes=['Move comparison state to CPU','Restore exact completed Task6 state once, proceed Task7 without repeating completed epochs'],old_train_sha256=hashlib.sha256(old.encode()).hexdigest(),new_train_sha256=hashlib.sha256(new.encode()).hexdigest(),failure_snapshot='check_readonly_20261025_000623',own_resume_and_stage7_admission='PENDING')
(R/'ENGINEERING02_PLAN.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
