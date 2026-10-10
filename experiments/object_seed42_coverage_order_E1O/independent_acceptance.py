"""External exit acceptance; read-only recomputation, no policy or environment."""
import os,sys,json,subprocess,time
from pathlib import Path
O=Path('/data/mlr_ifa/experiments/E1O_coverage_order_seed42_20261009')
S=Path('/data/mlr_ifa/experiment_workspaces/E1O_coverage_order_20261009/source')
assert (O/'pipeline.exit').read_text().strip()=='0'
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
residual=[p for p in processes if any(x in p for x in ['-m protective_e1.train','-m protective_e1.evaluate','protective_e1/pipeline.sh']) and 'grep' not in p]
assert not residual,residual
os.environ.update(E1_OUT=str(O),E1_SOURCE=str(S));sys.path.insert(0,str(S))
from protective_e1.finalize import main
main() # full input/source/weight hashes, epoch counts, index and episode replay
from protective_e1.io import write
result=json.loads((O/'FINAL_ACCEPTANCE.json').read_text())
assert result['training_epochs']==400 and result['new_optimizer_steps']==105500
assert result['formal_primary_rollouts']==200 and result['formal_secondary_rollouts']==680
result.update(status='COMPLETED',independent_acceptance='PASS',pipeline_exit=0,residual_processes=[],independent_checked_unix=time.time())
write(O/'INDEPENDENT_ACCEPTANCE.json',result);write(O/'RUN_STATUS.json',result)
with (O/'FINAL_REPORT.txt').open('a') as f:f.write('\nIndependent exit acceptance PASS; status COMPLETED.\n')
print(json.dumps(dict(status='COMPLETED',primary=result['e1o'],additional_seeds=result['additional_seeds'])))
