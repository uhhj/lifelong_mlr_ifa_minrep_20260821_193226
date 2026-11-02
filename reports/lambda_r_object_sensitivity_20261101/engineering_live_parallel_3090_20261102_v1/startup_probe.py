"""Zero-update isolated Python startup probe; current training is untouched."""
import os,sys,subprocess,json,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
D=Path('/data/mlr_ifa/diagnostics/live_parallel_startup_20261102_v1');assert not D.exists();D.mkdir()
envraw=(Path('/proc/4694/environ')).read_bytes().split(b'\0')
allowed={'PYTHONPATH','PYTHONNOUSERSITE','PYTHONSAFEPATH','PYTHONHOME'}
envfields={k.decode():v.decode() for item in envraw if b'=' in item for k,v in [item.split(b'=',1)] if k.decode() in allowed}
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for k in allowed:
 if k in envfields:env[k]=envfields[k]
 else:env.pop(k,None)
(D/'sitecustomize.py').write_text('import builtins\nbuiltins._isolated_startup_probe=True\n')
(D/'probe.py').write_text("import builtins,sys,json,site\nprint(json.dumps(dict(hook_loaded=getattr(builtins,'_isolated_startup_probe',False),path=sys.path,site=site.getsitepackages())))\n")
p=subprocess.run([sys.executable,'-B',str(D/'probe.py')],cwd=D,env=env,capture_output=True,text=True)
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
status=json.loads((Q/'L005_42/TRAIN_STATUS.json').read_text())
procs={}
for pid in [4347,8608,8609,4689,4694]:
 s=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split();procs[str(pid)]=dict(state=s[0],start=s[19])
out=dict(status='ZERO_UPDATE_STARTUP_PROBE',observed=time.time(),python=sys.executable,allowed_python_environment=envfields,probe_exit=p.returncode,probe_stdout=p.stdout,probe_stderr=p.stderr,processes=procs,train=status,gpu=gpu.stdout,mem_available_kib=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1]),optimizer_calls=0,environments=0,production_writes=0)
(D/'RESULT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
