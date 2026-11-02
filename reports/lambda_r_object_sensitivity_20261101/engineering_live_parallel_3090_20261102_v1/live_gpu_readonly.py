"""Bounded read-only GPU window during actual two-run production; no diagnostics."""
import json,time,subprocess,statistics
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
rows=[]
for _ in range(30):
 p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True);assert p.returncode==0
 rows.append(dict(time=time.time(),values=[float(x.strip()) for x in p.stdout.strip().split(',')]))
 time.sleep(1)
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True);assert p.returncode==0
print(json.dumps(dict(status='READ_ONLY_LIVE_TWO_RUN_GPU_WINDOW',rows=rows,mean_gpu_utilization=statistics.mean(r['values'][0] for r in rows),min_gpu_utilization=min(r['values'][0] for r in rows),max_gpu_utilization=max(r['values'][0] for r in rows),peak_used_MiB=max(r['values'][1] for r in rows),compute_processes=p.stdout,training={c:json.loads((Q/c/'TRAIN_STATUS.json').read_text()) for c in ['L005_42','L005_43']},optimizer_calls_added=0,environments_added=0,production_signals=0)))
