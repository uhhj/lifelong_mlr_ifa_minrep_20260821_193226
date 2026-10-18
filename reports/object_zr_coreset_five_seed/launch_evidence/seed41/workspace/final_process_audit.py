import os,json,subprocess,time
from pathlib import Path
O=Path('/root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed41_20261017')
assert(O/'pipeline.exit').read_text().strip()=='0'
audit=json.loads((O/'FINAL_AUDIT.json').read_text());assert audit['status']=='PASS_AWAITING_PROCESS_EXIT'
ps=subprocess.check_output(['ps','-eo','pid,args']).decode().splitlines()
remaining=[l for l in ps if any(x in l for x in ('-m zrg_exp.train','-m zrg_exp.evaluate','-m zrg_exp.finalize')) and 'ps -eo' not in l]
assert not remaining,remaining
(O/'PROCESS_EXIT_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS',pipeline_exit=0,remaining=remaining,unix=time.time()),indent=2))
(O/'RUN_STATUS.json').write_text(json.dumps(dict(status='COMPLETED',epochs=400,steps=105500,checkpoints=4,episodes=200,unix=time.time()),indent=2))
print('COMPLETED: independent process exit audit PASS')
