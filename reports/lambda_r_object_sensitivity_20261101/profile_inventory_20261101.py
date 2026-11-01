"""Read-only availability check, no debugger attachment or target signals."""
import json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
p=Path('/proc/2169')
assert str(root/'L005_41/train.py') in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
def read(p):
 try:return Path(p).read_text()
 except OSError as e:return type(e).__name__
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
found={n:shutil.which(n) for n in ['py-spy','perf','pidstat','iostat','nvidia-smi','gdb']}
extras=[str(p) for base in ['/data/mlr_ifa/env/bin','/data/mlr_ifa/tools','/usr/local/bin','/root/.local/bin'] for p in Path(base).glob('*spy*')]
print(json.dumps(dict(observed=time.time(),tools=found,extra_spy_paths=extras,uid=os.getuid(),ptrace_scope=read('/proc/sys/kernel/yama/ptrace_scope'),perf_paranoid=read('/proc/sys/kernel/perf_event_paranoid'),exe=str((p/'exe').resolve()),threads={t.name:dict(comm=read(t/'comm'),wchan=read(t/'wchan')) for t in (p/'task').iterdir()},train_status=json.loads((root/'L005_41/TRAIN_STATUS.json').read_text()),extra_updates=0,extra_environments=0,signals=0)))
