"""Read existing validation ledger and scientific state, never dispatch."""
import json,time,hashlib,os
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_r_branch_training_20261101_v1')
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 return dict(pid=pid,state=(p/'stat').read_text().split(')')[-1].split()[0],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
files={}
for n in ['DISPATCH.json','VALIDATION_PROGRESS.json','VALIDATION.json']:
 p=D/n
 if p.exists():
  raw=p.read_bytes();files[n]=dict(value=json.loads(raw),sha256=hashlib.sha256(raw).hexdigest())
pid=files['DISPATCH.json']['value']['pid']
print(json.dumps(dict(observed=time.time(),files=files,diagnostic_process=proc(pid),production_process=proc(2169),production_status=json.loads((Q/'L005_41/TRAIN_STATUS.json').read_text()),log_tail=(D/'validation.log').read_text(errors='replace')[-2000:],extra_updates_by_this_read=0,environments=0)))
