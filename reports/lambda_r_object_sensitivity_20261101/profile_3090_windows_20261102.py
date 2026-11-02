"""Three low-rate Python-only samples; never pause or inject into training."""
import json,time,subprocess,hashlib,os
from pathlib import Path
root=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
base=Path('/data/mlr_ifa/diagnostics/lambda_r_profile_20261101_v1')
tool=base/'py-spy-0.4.2';assert hashlib.sha256(tool.read_bytes()).hexdigest()=='9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594'
pid=2566;p=Path('/proc')/str(pid)
assert str(root/'engineering_resume_3090_20261102_v1/train_wrapper.py') in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
out=base/('sampling_3090_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir()
def snap():
 return dict(time=time.time(),stat=(p/'stat').read_text(),io=(p/'io').read_text(),schedstat=(p/'schedstat').read_text(),io_pressure=Path('/proc/pressure/io').read_text(),status=json.loads((root/'L005_41/TRAIN_STATUS.json').read_text()),epoch_history=json.loads((root/'L005_41/EPOCHS_S.json').read_text()))
windows=[]
for i in range(2):
 path=out/('window%d.txt'%i)
 args=[str(tool),'record','--pid',str(pid),'--nonblocking','--idle','--threads','--full-filenames','--rate','10','--duration','60','--format','raw','--output',str(path)]
 assert '--native' not in args and '--subprocesses' not in args and '--locals' not in args
 before=snap();start=time.time();q=subprocess.run(args,capture_output=True,text=True,timeout=80);after=snap()
 windows.append(dict(command=args,before=before,after=after,duration=time.time()-start,exit=q.returncode,stdout=q.stdout,stderr=q.stderr,stacks=path.read_text() if path.exists() else None))
 if q.returncode:break
print(json.dumps(dict(status='NONBLOCKING_PYTHON_STACK_SAMPLES_NOT_INSTRUMENTED_TIMING',windows=windows,profiler_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),output=str(out),extra_optimizer=0,extra_environment=0,training_code_or_state_written=False,signals_to_target=0,nonblocking=True,native=False,locals=False)))
