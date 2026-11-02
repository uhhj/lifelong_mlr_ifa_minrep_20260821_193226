"""Unique authorized post-reboot recovery dispatch. FILES supplied by reviewed local builder."""
import os,sys,json,time,hashlib,subprocess,shutil,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');E=Q/'engineering_resume_3090_20261102_v1';OLD=Q/'engineering_device_snapshot_20261101_v1'
D=Path('/data/mlr_ifa/diagnostics/lambda_recovery_3090_20261102_v2');PRIVATE=Path('/data/mlr_ifa/experiments')/Q.name
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not E.exists(),'Never repeat recovery dispatch'
lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert 'NVIDIA GeForce RTX 3090' in subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');state=(p/'stat').read_text().split(') ',1)[1].split()[0]
  assert int(p.name)==os.getpid() or state=='Z' or (str(Q) not in cmd and str(D) not in cmd),(p.name,cmd)
 except OSError:pass
assert not (Q/'controller.exit').exists() and not (OLD/'controller.exit').exists()
for c in [Q/'CONTRACT.json',OLD/'CONTRACT.json']:assert all(sha(Path(p))==h for p,h in js(c)['files'].items())
v=js(D/'VALIDATION.json');assert v['status']=='PASS_GPU_RECOVERY_AND_PAIRED_BRANCH_REPLAY' and v['started']==v['completed']==309 and v['extra_autograd']==11 and v['environments']==0 and v['production_metadata_unchanged']
assert all(x['all_gate_fields_exact'] and x['decision_equal'] for x in v['old_GPU_comparisons']) and len(v['old_GPU_comparisons'])==7
assert sha(D/'validate.py')=='52a310267a3ed2d534d85421d6f15d4a675e25559be428f81b558b15aeb7b0b5'
cp=PRIVATE/'L005_41/S/recovery.pth';assert sha(cp)=='65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070'
pause=js(Q/'user_pause_20261101_v1/STATUS.json');assert all(sha(Q/p)==h for p,h in pause['light_hashes'].items())
assert len(js(Q/'L005_41/EPOCHS_S.json'))==260
raw=(Q/'L005_41/GATES.jsonl').read_bytes();assert len(raw.splitlines())==4167
for case in ['L005_42','L005_43','L020_41','L020_42','L020_43']:assert not (Q/case/'train_S.log').exists()
free=os.statvfs('/data');assert free.f_bfree*free.f_frsize>12*1024**3+3*1024**3
E.mkdir();backupdir=PRIVATE/'engineering_resume_3090_20261102_v1';backupdir.mkdir()
backup=backupdir/'interrupted_epoch260.pth';shutil.copyfile(cp,backup);assert sha(backup)==sha(cp)
for name,source in FILES.items():(E/name).write_text(source)
(E/'VALIDATION.json').write_bytes((D/'VALIDATION.json').read_bytes())
(E/'FROZEN_TRAIN_INPUT_8.json').write_bytes((Q/'L005_41/TRAIN_INPUT_S_8.json').read_bytes())
(E/'INTERRUPTED_GATES.jsonl').write_bytes(raw)
(E/'ORIGINAL_EPOCHS.json').write_bytes((Q/'L005_41/EPOCHS_S.json').read_bytes())
interruption=dict(status='AUTHORIZED_3090_CHECKPOINT_RECOVERY',time=time.time(),checkpoint_sha256=sha(cp),checkpoint_backup=str(backup),original_epochs=260,original_gates=4167,checkpoint_gates=4160,recorded_partial_optimizer_calls=140,unrecorded_partial_cost='UNKNOWN',extra_recovery_diagnostic_optimizer_calls=309,extra_recovery_diagnostic_autograd=11,prior_engineering_optimizer_calls=7828,prior_engineering_autograd=36,first_failed_validation_optimizer_calls=0,first_failed_validation_autograd=0,first_failed_validation_source='lambda_recovery_3090_20261102_v1',gpu='RTX3090',previous_gpu='RTX4080SUPER',user_authorized_cross_hardware=True,completed_epochs_repeated=0,partial_reconstruction='Same7gate fields verified; original lines retained, matching reconstructed records not appended again',scientific_acceptance='PENDING',cross_hardware_full_run_exact_claim=False)
(E/'INTERRUPTION.json').write_text(json.dumps(interruption,indent=2))
names=[*FILES,'VALIDATION.json','FROZEN_TRAIN_INPUT_8.json','INTERRUPTED_GATES.jsonl','ORIGINAL_EPOCHS.json','INTERRUPTION.json']
c=dict(version=E.name,scientific_contract_sha256=sha(Q/'CONTRACT.json'),prior_engineering_contract_sha256=sha(OLD/'CONTRACT.json'),files={str(E/n):sha(E/n) for n in names},completion_source=str(E/'controller.exit'),authorization='2026-11-02 user permitted verified RTX3090 recovery',optimizer_diagnostic_calls=309,extra_autograd=11,production_budget_unchanged=True,hardware_comparability_limitation=True)
(E/'CONTRACT.json').write_text(json.dumps(c,indent=2));fcntl.flock(lock,fcntl.LOCK_UN)
with (E/'controller.log').open('x') as f:
 p=subprocess.Popen([sys.executable,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
out=dict(status='RECOVERY_CONTROLLER_DISPATCHED_NOT_FINAL',pid=p.pid,time=time.time(),contract_sha256=sha(E/'CONTRACT.json'),contract=c,interruption=interruption)
(E/'DISPATCH.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
