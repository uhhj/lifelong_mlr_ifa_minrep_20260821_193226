"""Small read-only resource samples; no experiment, optimizer or environment calls."""
import json,subprocess,time,os
from pathlib import Path
ROOT=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
samples=[]
for i in range(3):
 samples.append(dict(time=time.time(),gpu=cmd(['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate','--format=csv,noheader']),gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),load=Path('/proc/loadavg').read_text(),meminfo=Path('/proc/meminfo').read_text(),own_processes=cmd(['ps','-p','2163,2164,2169','-o','pid,ppid,stat,pcpu,pmem,rss,etime,time,comm']),proc_io={p:(Path('/proc')/p/'io').read_text() for p in ['2169'] if (Path('/proc')/p/'io').exists()}))
 if i<2:time.sleep(5)
print(json.dumps(dict(status='READONLY_RESOURCE_SAMPLES_NOT_BOTTLENECK_PROOF',observed=time.time(),root=str(ROOT),samples=samples,gpu_details=cmd(['nvidia-smi','-q','-d','PERFORMANCE,CLOCK,POWER']),optimizer_calls=0,environments=0,remote_writes=0,limitation='Short GPU samples and process averages do not establish a sustained bottleneck or explain cross-run resource variation.')))
