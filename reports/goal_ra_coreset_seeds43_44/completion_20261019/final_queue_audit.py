"""No-model process-exit acceptance after finite queue has returned."""
from pathlib import Path
import json,subprocess,time,os,hashlib
Q=Path(__file__).resolve().parent
assert (Q/'queue.exit').read_text().strip()=='0'
pid=(Q/'queue.pid').read_text().strip();assert not (Path('/proc')/pid).exists()
procs=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
bad=[x for x in procs if ' -m coreset_exp.' in x and any(t in x for t in ['train','evaluate'])];assert not bad,bad
for seed in (43,44):
 O=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_20261019'%seed)
 assert json.loads((O/'PROCESS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
 assert (O/'supervisor.exit').read_text().strip()=='0'
for n,h in json.loads((Q/'ORCHESTRATION_SHA256.json').read_text()).items():assert hashlib.sha256((Q/n).read_bytes()).hexdigest()==h,n
a=json.loads((Q/'THREE_SEED_RESULTS.json').read_text());assert a['RA_formal_records']==600 and a['optimizer_steps']==228300
out=dict(status='PASS',time=time.time(),queue_exit=0,no_remaining_scientific_processes=True,RA_records=600,RA_checkpoints=12,epochs=1200,optimizer_steps=228300,aggregate_sha256=hashlib.sha256((Q/'THREE_SEED_RESULTS.json').read_bytes()).hexdigest())
(Q/'FINAL_QUEUE_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
(Q/'RUN_STATUS.json').write_text(json.dumps(dict(out,status='COMPLETED_PENDING_LOCAL_RECOMPUTATION'),indent=2)+'\n');print(json.dumps(out))
