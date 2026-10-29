"""Bounded read-only system inventory. No CUDA initialization, signals, or training."""
import os,json,time,shutil,subprocess,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')
def run(args):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=15);return dict(rc=r.returncode,out=r.stdout[-16000:],err=r.stderr[-2000:])
 except Exception as e:return dict(error=type(e).__name__+': '+str(e))
def read(p):
 try:return Path(p).read_text()
 except OSError as e:return type(e).__name__
c=json.loads((Q/'CONTRACT.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(Q) in cmd and int(p.name)!=os.getpid():procs.append(dict(pid=int(p.name),cmd=cmd,status=read(p/'status'),cgroup=read(p/'cgroup'),io=read(p/'io'),wchan=read(p/'wchan')))
 except OSError:pass
print(json.dumps(dict(status='READONLY_PERFORMANCE_INVENTORY',time=time.time(),contract_sha256=hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest(),procs=procs,tools={x:shutil.which(x) for x in ['py-spy','perf','pidstat','iostat','nvidia-smi']},python_version=run(['/data/mlr_ifa/env/bin/python','--version']),packages=run(['/data/mlr_ifa/env/bin/python','-m','pip','show','py-spy']),gpu=run(['nvidia-smi','-q']),cpuinfo=read('/proc/cpuinfo')[:2500],loadavg=read('/proc/loadavg'),pressure={x:read('/proc/pressure/'+x) for x in ['cpu','io','memory']},cgroup={x:read('/sys/fs/cgroup/'+x) for x in ['cpu.max','cpu.stat','cpu.pressure','cpuset.cpus.effective','memory.max','memory.current']},ptrace_scope=read('/proc/sys/kernel/yama/ptrace_scope'),environment_changes=0,optimizer_calls=0,environments=0)))
