"""Read-only /proc+cgroup+GPU sampling. Never signals or attaches to training."""
import os,json,time,subprocess,hashlib
from pathlib import Path
root=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
def read(p):
 try:return Path(p).read_text()
 except (OSError,UnicodeError) as e:return 'UNAVAILABLE:'+type(e).__name__
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
pid=2169;p=Path('/proc')/str(pid)
assert str(root/'L005_41/train.py') in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
cg=read(p/'cgroup');paths={Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup/cpu'),Path('/sys/fs/cgroup/cpu,cpuacct'),Path('/sys/fs/cgroup/cpuacct')}
for line in cg.splitlines():
 parts=line.split(':',2)
 if len(parts)!=3:continue
 hierarchy,controllers,relative=parts
 for base in [Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup')/controllers,*[Path('/sys/fs/cgroup')/x for x in controllers.split(',') if x]]:
  candidate=base/relative.lstrip('/')
  if candidate.exists() and str(candidate.resolve()).startswith('/sys/fs/cgroup'):
   paths.add(candidate)
   for ancestor in candidate.parents:
    if str(ancestor).startswith('/sys/fs/cgroup'):paths.add(ancestor)
cg_names=['cpu.stat','cpu.max','cpu.weight','cpu.pressure','cpu.cfs_quota_us','cpu.cfs_period_us','cpu.shares','cpuacct.usage','cpuset.cpus','cpuset.cpus.effective','memory.events','memory.pressure','io.pressure']
cg_files=sorted({str(x/n) for x in paths for n in cg_names if (x/n).is_file()})
samples=[]
for i in range(31):
 sample=dict(time=time.time(),stat=read(p/'stat'),schedstat=read(p/'schedstat'),io=read(p/'io'),status=read(p/'status'),cgroups={n:read(n) for n in cg_files},host_cpu=read('/proc/stat').splitlines()[:1],pressure={n:read('/proc/pressure/'+n) for n in ['cpu','io','memory']},vmstat=read('/proc/vmstat'),threads={x.name:dict(stat=read(x/'stat'),schedstat=read(x/'schedstat'),wchan=read(x/'wchan')) for x in (p/'task').iterdir()})
 if i%5==0:sample['gpu']=cmd(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm','--format=csv,noheader'])
 samples.append(sample)
 if i<30:time.sleep(1)
hist=root/'L005_41/EPOCHS_S.json';raw=hist.read_bytes();history=json.loads(raw)
cpuinfo=[x for x in read('/proc/cpuinfo').splitlines() if x.startswith(('processor','model name','cpu MHz','cpu cores','siblings'))]
print(json.dumps(dict(status='PASS_READONLY_SLOWDOWN_OBSERVATION_NOT_CAUSAL_PROFILE',observed=time.time(),pid=pid,clock_ticks=os.sysconf('SC_CLK_TCK'),cgroup_path=cg,cgroup_files=cg_files,affinity=sorted(os.sched_getaffinity(pid)),cpu_count=os.cpu_count(),cpuinfo=cpuinfo,meminfo=read('/proc/meminfo'),mounts=read('/proc/mounts'),samples=samples,history=history,history_sha256=hashlib.sha256(raw).hexdigest(),optimizer_calls=0,environments=0,remote_writes=0,signals_sent=0,profiler_attached=False)))
