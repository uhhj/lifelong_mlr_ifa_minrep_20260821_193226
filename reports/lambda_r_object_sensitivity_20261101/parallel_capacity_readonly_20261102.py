"""Read available resources for feasibility only; never start another training job."""
import json,os,time,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
def read(p):
 p=Path(p)
 return p.read_text() if p.exists() else None
def run(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and str(Q) in cmd:
   lines=(p/'status').read_text().splitlines()
   processes.append(dict(pid=int(p.name),cmd=cmd,status=[x for x in lines if x.split(':')[0] in ['State','VmRSS','VmHWM','Threads','Cpus_allowed_list','Mems_allowed_list']],cgroup=(p/'cgroup').read_text()))
 except OSError:pass
print(json.dumps(dict(status='READONLY_PARALLEL_CAPACITY_NOT_BENCHMARK',observed=time.time(),cpu_count=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)),load=read('/proc/loadavg'),memory=read('/proc/meminfo'),cgroup=read('/proc/self/cgroup'),limits={p:read(p) for p in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu.stat','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/memory/memory.limit_in_bytes']},gpu=run(['nvidia-smi','--query-gpu=name,compute_mode,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),gpu_processes=run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),processes=processes,queue=json.loads((Q/'STATUS.json').read_text()),optimizer_calls=0,environments=0,signals=0,production_writes=0)))
