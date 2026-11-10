from pathlib import Path
import sys,json,time,subprocess,hashlib,concurrent.futures
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(E));from resume_runtime import verify,processes,P
verify()
tool=Path('/data/mlr_ifa/diagnostics/lambda_r_profile_20261101_v1/py-spy-0.4.2')
assert hashlib.sha256(tool.read_bytes()).hexdigest()=='9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594'
pid=40723;p=Path('/proc')/str(pid)
assert str(E/'evaluate_guarded.py') in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
D=Path('/data/mlr_ifa/diagnostics')/('align_eval_profile_'+time.strftime('%Y%m%d_%H%M%S'));D.mkdir()
def gpu():
 rows=[]
 for i in range(40):
  r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
  rows.append(dict(unix=time.time(),exit=r.returncode,text=r.stdout.strip()));time.sleep(1)
 return rows
before=dict(unix=time.time(),stat=(p/'stat').read_text(),io=(p/'io').read_text(),pressure={x:Path('/proc/pressure',x).read_text() for x in ['io','cpu','memory']})
args=[str(tool),'record','--pid',str(pid),'--nonblocking','--rate','20','--duration','40','--format','raw','--full-filenames','--output',str(D/'stack.txt')]
with concurrent.futures.ThreadPoolExecutor(1) as ex:
 future=ex.submit(gpu);r=subprocess.run(args,capture_output=True,text=True,timeout=55);samples=future.result()
after=dict(unix=time.time(),stat=(p/'stat').read_text(),io=(p/'io').read_text())
verify()
print(json.dumps(dict(status='READONLY_NONBLOCKING_STACK_PROFILE',command=args,exit=r.returncode,stdout=r.stdout,stderr=r.stderr,stacks=(D/'stack.txt').read_text() if (D/'stack.txt').exists() else None,gpu=samples,before=before,after=after,processes=processes([Q,P]),new_optimizer=0,new_environment=0,signals=0,production_files_modified=False)))
