"""120s nonblocking external stack and OS sampling; never executes model updates."""
import os,time,json,hashlib,subprocess,ast,re
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')
D=Path('/data/mlr_ifa/diagnostics/support_gate_performance_20261029_v1')
def read(p):
 try:return Path(p).read_text()
 except OSError as e:return type(e).__name__
def js(p):return json.loads(Path(p).read_text())
c=js(Q/'CONTRACT.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
targets=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if str(Q) in cmd and '/train.py S' in cmd and 'torch.distributed.run' not in cmd:targets.append((int(p.name),cmd))
 except OSError:pass
assert len(targets)==1,targets
pid,cmd=targets[0];seed=int(re.search(r'/seed(\d+)/train.py',cmd).group(1));P=Path('/proc')/str(pid)
started=time.time();outpath=D/('stacks_%d.raw'%int(started));assert not outpath.exists()
before=js(Q/f'seed{seed}/EPOCHS_S.json')
args=[str(D/'py-spy-0.4.1'),'record','--pid',str(pid),'--duration','120','--rate','50','--nonblocking','--idle','--threads','--full-filenames','--format','raw','--output',str(outpath)]
spy=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
samples=[]
for i in range(60):
 t=time.monotonic();gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 samples.append(dict(time=time.time(),proc_stat=read(P/'stat'),schedstat=read(P/'schedstat'),io=read(P/'io'),wchan=read(P/'wchan'),syscall=read(P/'syscall'),cpu_stat=read('/proc/stat').splitlines()[:13],pressure={x:read('/proc/pressure/'+x) for x in ['cpu','io','memory']},cpu_quota=read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),cpu_cgroup_stat=read('/sys/fs/cgroup/cpu/cpu.stat'),gpu=dict(rc=gpu.returncode,out=gpu.stdout,err=gpu.stderr)))
 time.sleep(max(0,2-(time.monotonic()-t)))
stdout,stderr=spy.communicate(timeout=20)
after=js(Q/f'seed{seed}/EPOCHS_S.json');assert after[:len(before)]==before
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source')
source={}
for file,names in [('caet_v3_exp/train.py',['cpu']),('coreset_exp/common.py',['save']),('libero/lifelong/algos/mlr.py',['FileTrajectoryDataset'])]:
 p=S/file
 if p.exists():
  text=p.read_text();tree=ast.parse(text);nodes=[n for n in tree.body if getattr(n,'name',None) in names]
  source[file]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),definitions=[ast.get_source_segment(text,n) for n in nodes])
result=dict(status='NONBLOCKING_SAMPLING_NO_SCIENTIFIC_MUTATION',started=started,ended=time.time(),pid=pid,seed=seed,command=cmd,profiler_argv=args,profiler_exit=spy.returncode,profiler_stdout=stdout,profiler_stderr=stderr,stacks=read(outpath) if outpath.exists() else None,clock_ticks=os.sysconf('SC_CLK_TCK'),samples=samples,epochs_before=len(before),epochs_after=len(after),new_epoch_rows=after[len(before):],sources=source,frozen_files_exact=True,prefix_exact=True,optimizer_calls_added_by_diagnostic=0,environments_added=0,signals_to_target=0,limitations=['Asynchronous nonblocking stacks can be incomplete; sample percentages are wall-stack occupancy, not CUDA kernel elapsed timings.','Profiler runs outside the training interpreter; OS queries add small observational overhead.'])
(D/('SAMPLE_%d.json'%int(started))).write_text(json.dumps(result));print(json.dumps(result))
