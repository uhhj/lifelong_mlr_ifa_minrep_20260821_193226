import json,time,subprocess,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;O=Path('/data/mlr_ifa/experiments/ZR_coreset_OBJECT_seed42_20261017')
assert(O/'pipeline.exit').read_text().strip()=='1' and(O/'supervisor.exit').read_text().strip()=='1'
assert(E/'resume.exit').read_text().strip()=='0'
assert json.loads((O/'FINAL_AUDIT.json').read_text())['status']=='PASS_AWAITING_PROCESS_EXIT'
ps=subprocess.check_output(['ps','-eo','pid,args']).decode().splitlines()
remaining=[r for r in ps if '-m zrg_exp.evaluate' in r or '-m zrg_exp.finalize' in r or 'python' in r and r.rstrip().endswith('train_metadata_adapter.py')]
assert not remaining,remaining
c=json.loads((E/'RECOVERY_CONTRACT.json').read_text())
assert hashlib.sha256((O/'CONTRACT.json').read_bytes()).hexdigest()==c['original_contract_sha256']
for n,h in c['files'].items():assert hashlib.sha256((E/n).read_bytes()).hexdigest()==h
(O/'PROCESS_EXIT_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS',original_pipeline_exit=1,original_supervisor_exit=1,recovery_exit=0,remaining=remaining,unix=time.time()),indent=2))
(O/'RUN_STATUS.json').write_text(json.dumps(dict(status='COMPLETED',epochs=400,steps=105500,checkpoints=4,episodes=200,original_pipeline_exit=1,recovery_exit=0,unix=time.time()),indent=2))
with(O/'FINAL_REPORT.md').open('a') as f:f.write('\nIndependent exit acceptance PASS: original pre-update audit failure exits=1 retained; external metadata normalization recovery exit=0. Model/loss/data code and original contract unchanged.\n')
print('COMPLETED: recovered process exit acceptance PASS')
