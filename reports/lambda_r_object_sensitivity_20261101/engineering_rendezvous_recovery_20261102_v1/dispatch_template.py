"""Unique versioned recovery deployment after actual launcher and GPU replay admission."""
import os,sys,json,time,hashlib,subprocess,shutil
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');E=Q/'engineering_rendezvous_recovery_20261102_v1';D=Path('/data/mlr_ifa/diagnostics/rendezvous_recovery_20261102_v1');R=Q/'L005_43';P=Path('/data/mlr_ifa/experiments')/Q.name/'L005_43/S';js=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES={}
assert set(SOURCES)=={'controller.py','core_wrapper.py','train_wrapper.py','checks_adapter.py','resume_current.py'}
assert not E.exists(),'Never repeat recovery deployment'
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  assert not (str(Q) in cmd or str(D) in cmd) or int(p.name)==os.getpid(),cmd
 except OSError:pass
for path in [Q/'CONTRACT.json',*Q.glob('engineering_*/CONTRACT.json')]:assert all(sha(p)==h for p,h in js(path)['files'].items())
v=js(D/'VALIDATION.json');b=js(D/'BINDING.json');lv=js(D/'LAUNCHER_VALIDATION.json')
assert v['status']=='PASS_SAME_GPU_RECOVERY_PARTIAL_PREFIX_AND_NEXT_BLOCK' and v['started']==v['completed']==869 and v['extra_autograd']==11 and v['environments']==0
assert v['full_state_restore_exact'] and v['save_load_roundtrip_exact'] and v['production_metadata_unchanged'] and v['restored_after_diagnostics'] and len(v['comparisons'])==21
assert all(x['all_candidate_rows_states_probes_decisions_exact'] for x in v['comparisons']) and all(x['original_gate_exact'] for x in v['comparisons'][:20]) and v['comparisons'][-1]['next_unsaved_block']
assert lv['status']=='PASS_INDEPENDENT_EPHEMERAL_RENDEZVOUS_LIFECYCLE' and lv['early_peer_exit_survived'] and lv['optimizer_calls']==lv['environments']==0
assert all(sha(D/n)==h for n,h in b['source_hashes'].items()) and all(sha(p)==h for p,h in b['production_metadata'].items())
assert sha(P/'recovery.pth')==sha(b['backup'])==v['checkpoint_sha256']=='6458bc06f8e99766d360faba812f1c56aff2962616b92931a1806663a750f222'
assert len(js(R/'EPOCHS_S.json'))==289 and len((R/'GATES.jsonl').read_text().splitlines())==5079
assert os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize>3*1024**3
for case in ['L005_41','L005_42']:
 assert (Q/case/'train_S.exit').read_text().strip()=='0' and len(js(Q/case/'EPOCHS_S.json'))==400
for case in ['L020_41','L020_42','L020_43']:assert not (Q/case/'train_S.log').exists()
for case in ['L005_41','L005_42','L005_43']:assert not (Q/case/'evaluate.log').exists()
assert (Q/'engineering_live_parallel_3090_20261102_v1/HOOK_ENABLED.json').exists()
failures=['L005_43/train_S.exit','engineering_live_parallel_3090_20261102_v1/controller.exit','engineering_live_parallel_3090_20261102_v1/FAILURE.json'];assert all((Q/n).exists() for n in failures)
completed={}
for case in ['L005_41','L005_42']:
 completed.update({str(p):sha(p) for p in (Q/case).glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']})
 for st in range(6,10):
  p=Path('/data/mlr_ifa/experiments')/Q.name/case/'S'/('stage%d.pth'%st);completed[str(p)]=sha(p)
E.mkdir()
for n,s in SOURCES.items():(E/n).write_text(s)
for source,target in [(D/'VALIDATION.json','VALIDATION.json'),(D/'LAUNCHER_VALIDATION.json','LAUNCHER_VALIDATION.json'),(D/'BINDING.json','VALIDATION_BINDING.json'),(R/'EPOCHS_S.json','ORIGINAL_EPOCHS.json'),(R/'GATES.jsonl','INTERRUPTED_GATES.jsonl'),(R/'TRAIN_INPUT_S_8.json','FROZEN_TRAIN_INPUT_8.json'),(Q/'engineering_resume_3090_20261102_v1/L005_43_BRANCH_COPY_EQ.json','ORIGINAL_BRANCH_COPY_EQ.json'),(Q/'engineering_checks_3090_20261102_v1/L005_43_CHECK_EQ.json','ORIGINAL_CHECK_EQ.json')]:shutil.copyfile(source,E/target)
(E/'MITIGATED_FAILURES.json').write_text(json.dumps({n:sha(Q/n) for n in failures},indent=2))
(E/'INTERRUPTION.json').write_text(json.dumps(dict(history_epochs=289,gate_blocks_complete_epochs=5059,recorded_partial_blocks=20,recorded_partial_optimizer=400,unrecorded_partial_cost='UNKNOWN',previous_partial_optimizer=140,total_recorded_abandoned_partial_optimizer=540,completed_hashes=completed,checkpoint_backup=b['backup'],checkpoint_sha256=v['checkpoint_sha256'],original_log_sha256=sha(R/'train_S.log'),frozen_scientific_contract=sha(Q/'CONTRACT.json')),indent=2))
names=[*SOURCES,'VALIDATION.json','LAUNCHER_VALIDATION.json','VALIDATION_BINDING.json','ORIGINAL_EPOCHS.json','INTERRUPTED_GATES.jsonl','FROZEN_TRAIN_INPUT_8.json','ORIGINAL_BRANCH_COPY_EQ.json','ORIGINAL_CHECK_EQ.json','MITIGATED_FAILURES.json','INTERRUPTION.json']
c=dict(version=E.name,files={str(E/n):sha(E/n) for n in names},scientific_contract_sha256=sha(Q/'CONTRACT.json'),prior_live_contract_sha256=sha(Q/'engineering_live_parallel_3090_20261102_v1/CONTRACT.json'),diagnostic_optimizer_calls=35418,diagnostic_extra_autograd=94,recovery_optimizer_calls=869,recovery_extra_autograd=11,environments_added=0,production_signals=0,recorded_abandoned_partial_optimizer=540,unrecorded_abandoned_partial_cost='UNKNOWN',change='Independent ephemeral rendezvous per launcher; exact checkpoint/partial-prefix continuation of L00543; no science change')
(E/'CONTRACT.json').write_text(json.dumps(c,indent=2))
with (E/'controller.log').open('x') as f:p=subprocess.Popen([sys.executable,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
r=dict(status='RECOVERY_DISPATCHED_PENDING_PRODUCTION_AUDIT',pid=p.pid,time=time.time(),contract_sha256=sha(E/'CONTRACT.json'),contract=c)
(E/'DISPATCH.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
