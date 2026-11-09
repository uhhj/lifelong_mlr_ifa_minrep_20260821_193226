"""Twenty one-second nvidia-smi samples; no training mutation."""
import subprocess,time,json,datetime
from pathlib import Path
rows=[]
query=['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu','--format=csv,noheader,nounits']
for i in range(20):
 p=subprocess.run(query,capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 cols=[v.strip() for v in p.stdout.strip().split(',')]
 rows.append(dict(time=time.time(),name=cols[0],gpu_percent=int(cols[1]),memory_controller_percent=int(cols[2]),memory_used_MiB=int(cols[3]),memory_total_MiB=int(cols[4]),power_W=float(cols[5]),power_limit_W=float(cols[6]),temperature_C=int(cols[7])))
 if i<19:time.sleep(1)
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
root=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT41_42_43_20261109_rerun_v1')
status={}
for seed in [41,42,43]:
 f=root/('Q1_'+str(seed))/'TRAIN_STATUS.json'
 if f.exists():status[str(seed)]=json.loads(f.read_text())
print(json.dumps(dict(status='READ_ONLY_GPU_WINDOW_NOT_FINAL_ACCEPTANCE',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),samples=rows,summary=dict(samples=len(rows),mean_gpu_percent=sum(r['gpu_percent'] for r in rows)/len(rows),min_gpu_percent=min(r['gpu_percent'] for r in rows),max_gpu_percent=max(r['gpu_percent'] for r in rows),last=rows[-1]),compute_processes=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr),training_status=status,optimizer_calls=0,environments=0)))
