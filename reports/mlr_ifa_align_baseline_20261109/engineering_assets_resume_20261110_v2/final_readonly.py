from pathlib import Path
import sys,json,time,subprocess,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(E));from resume_runtime import *
verify();assert (E/'controller.exit').read_text().strip()=='0' and not processes([Q,P])
assert js(E/'RNG_BOUNDARY_AUDIT.json')['status']=='PASS_EXACT_PRE_CELL_RNG'
with (E/'FINAL_CPU_REQUEST.json').open('x') as f:json.dump(dict(status='RUNNING_NO_REPEAT',observed=time.time()),f)
out=E/('final_cpu_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();results={}
for case in CASES:
 assert (E/(case+'_evaluate.exit')).read_text().strip()=='0'
 en=env(case);en['CUDA_VISIBLE_DEVICES']='';en['OMP_NUM_THREADS']='1'
 p=subprocess.run([PY,'-B',str(Q/'final_case_cpu.py'),case],env=en,cwd=Q,capture_output=True,text=True)
 write(out/(case+'_ATTEMPT.json'),dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,(case,p.stderr[-2000:])
 r=json.loads(p.stdout.strip().splitlines()[-1]);assert r['status']=='PASS_CPU_CASE_RECONSTRUCTION';results[case]=r
verify();assert not processes([Q/'cases',P])
write(out/'CPU_RESULTS.json',dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cases=results,optimizer_calls=0,environments=0))
scope={'__name__':'final_readonly_snapshot'}
# The frozen recovery check emits the actual old exit1 plus actual new exit0; never synthesize success.
import contextlib,io
buf=io.StringIO()
with contextlib.redirect_stdout(buf):exec(compile((E/'check_readonly.py').read_text(),str(E/'check_readonly.py'),'exec'),scope)
s=json.loads(buf.getvalue());s.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cpu=results,remote_cpu_directory=str(out));print(json.dumps(s))
