import os,json,hashlib,time,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1')
P=Path('/data/mlr_ifa/experiments')/Q.name
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
assert (Q/'controller.exit').read_text()=='1' and (Q/'GOAL_41/train_Q.exit').read_text()=='1'
log=(Q/'GOAL_41/train_Q.log').read_text();assert "has no attribute 'total_steps'" in log and "loader,sampler=setup(a,ds)" in log
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 if int(p.name)!=os.getpid() and str(Q) in cmd:processes.append(cmd)
assert not processes
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for pat in ['EPOCHS_*.json','ADMISSION_*.json','SCHEDULE_*.json','COLLECTION_STARTED_*.json','OUTCOMES.json','EVALUATION_STARTED.json']:
 assert not list(Q.rglob(pat)),pat
assert not list(P.rglob('*.pth'))
print(json.dumps(dict(status='PASS_FAILED_BEFORE_FIRST_OPTIMIZER_OR_ENVIRONMENT',observed=time.time(),root=str(Q),contract_sha256=sha(Q/'CONTRACT.json'),failure_sha256=sha(Q/'FAILURE.json'),log_sha256=sha(Q/'GOAL_41/train_Q.log'),initial_sha256=sha(Q/'GOAL_41/INITIAL_Q.json'),phase=json.loads((Q/'GOAL_41/TRAIN_STATUS.json').read_text()),processes=processes,optimizer_calls=0,environment_calls=0,complete_epochs=0,recovery_files=0,failure_location='setup scheduler validation; before stage admission or update; frozen start_task only creates optimizer/scheduler',preserved_files=[str(p.relative_to(Q)) for p in Q.rglob('*') if p.is_file()])))
