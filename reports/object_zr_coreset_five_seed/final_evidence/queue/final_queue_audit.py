"""Run after queue exit, never while still training or evaluating."""
import hashlib,json,subprocess,time
from pathlib import Path
Q=Path(__file__).resolve().parent
assert (Q/'queue.exit').read_text().strip()=='0'
for p,h in json.loads((Q/'ORCHESTRATION_SHA256.json').read_text()).items():
 assert hashlib.sha256((Q/p).read_bytes()).hexdigest()==h,p
ps=subprocess.check_output(['ps','-eo','pid,args']).decode().splitlines()
remaining=[x for x in ps if any(k in x for k in ('-m zrg_exp.train','-m zrg_exp.evaluate','-m zrg_exp.finalize','bash '+str(Q/'run_queue.sh'))) and 'ps -eo' not in x]
assert not remaining,remaining
for seed in (40,41):
 o=Path(f'/root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed{seed}_20261017')
 assert json.loads((o/'RUN_STATUS.json').read_text())['status']=='COMPLETED'
 assert (o/'pipeline.exit').read_text().strip()=='0' and (o/'supervisor.exit').read_text().strip()=='0'
s=json.loads((Q/'FIVE_SEED_SUMMARY.json').read_text());assert s['new_extension_episodes']==400
out=dict(status='PASS_PENDING_LOCAL_RECOMPUTATION',queue_exit=0,no_residual_process=True,completed_42_43_44_readonly=True,new_epochs=800,new_steps=211000,new_final_checkpoints=8,new_episodes=400,five_seed_new_records=1000,unix=time.time())
(Q/'FINAL_QUEUE_ACCEPTANCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
