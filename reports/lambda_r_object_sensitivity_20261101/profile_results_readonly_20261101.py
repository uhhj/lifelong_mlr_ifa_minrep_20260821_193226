"""Recover already-completed diagnostic JSON; never rerun the benchmark."""
import json,hashlib,time
from pathlib import Path
p=Path('/data/mlr_ifa/diagnostics/lambda_r_branch_copy_benchmark_20261101_v1/RESULT.json')
raw=p.read_bytes()
q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
procs=[]
for t in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(t/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if str(q) in cmd:procs.append(dict(pid=int(t.name),cmd=cmd,state=(t/'stat').read_text().split(')')[-1].split()[0]))
 except OSError:pass
print(json.dumps(dict(observed=time.time(),result=json.loads(raw),result_sha256=hashlib.sha256(raw).hexdigest(),processes=procs,train_status=json.loads((q/'L005_41/TRAIN_STATUS.json').read_text()),extra_optimizer=0,extra_environment=0)))
