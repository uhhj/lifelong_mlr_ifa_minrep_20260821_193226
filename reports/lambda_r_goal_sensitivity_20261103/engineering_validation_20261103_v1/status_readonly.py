"""Read the existing GOAL validation only; never dispatch or repeat it."""
from pathlib import Path
import os,json,hashlib,time
D=Path('/data/mlr_ifa/diagnostics/lambda_r_goal_engineering_20261103_v1')
assert D.exists()
files={}
for n in ['DISPATCH.json','VALIDATION_PROGRESS.json','VALIDATION.json','LOCK_VALIDATION.json','validation.exit','INPUT_BINDING.json']:
 p=D/n
 if p.exists():files[n]=dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and str(D) in cmd:processes.append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0]))
 except OSError:pass
print(json.dumps(dict(status='READ_ONLY_GOAL_VALIDATION_LEDGER',observed=time.time(),directory=str(D),files=files,processes=processes,log_tail=(D/'validation.log').read_text(errors='replace')[-6000:])))
