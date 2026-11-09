import subprocess,time,json,os,statistics
from pathlib import Path
samples=[]
for i in range(30):
 p=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate','--format=csv,noheader,nounits'],capture_output=True,text=True)
 samples.append(dict(time=time.time(),exit=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip()))
 if i<29:time.sleep(1)
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
capacity={}
for m in ['/','/data']:
 v=os.statvfs(m);capacity[m]=dict(total=v.f_blocks*v.f_frsize,available=v.f_bavail*v.f_frsize)
print(json.dumps(dict(status='READONLY_GPU_SAMPLE',observed=time.time(),samples=samples,compute_processes=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr),capacity=capacity)))
