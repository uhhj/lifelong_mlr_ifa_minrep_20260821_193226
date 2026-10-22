import os,json,time,hashlib
from pathlib import Path
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source')
out=dict(observed_unix=time.time(),processes=[],base_source=(S/'libero/lifelong/algos/base.py').read_text(),server_writes=0,optimizer_calls=0,environment_steps=0)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and 'CPU_AUTOGRAD_AND_ACTUAL_GMM_IDENTITY_PASS' in cmd:
   out['processes'].append(dict(pid=int(p.name),command_sha256=hashlib.sha256(cmd.encode()).hexdigest(),wchan=(p/'wchan').read_text(),stat=(p/'stat').read_text(),status=(p/'status').read_text()))
 except OSError:pass
print(json.dumps(out))
