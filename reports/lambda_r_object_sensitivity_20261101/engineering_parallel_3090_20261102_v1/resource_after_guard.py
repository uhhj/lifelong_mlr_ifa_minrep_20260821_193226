"""Read-only resource observation after diagnostic headroom guard."""
import json,time,subprocess
from pathlib import Path
mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');root=json.loads((q/'STATUS.json').read_text());train=json.loads((q/'L005_42/TRAIN_STATUS.json').read_text())
print(json.dumps(dict(observed=time.time(),memory_kib=mem,gpu=dict(exit=gpu.returncode,stdout=gpu.stdout),gpu_processes=apps.stdout,root=root,train=train,optimizer_calls=0,environments=0)))
