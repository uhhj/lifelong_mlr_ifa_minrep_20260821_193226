from pathlib import Path
import sys,json,time,subprocess,io,contextlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_cached_eval_20261110_v1'
sys.path.insert(0,str(E));from runtime_cached import *
verify();assert (E/'controller.exit').read_text().strip()=='0' and not processes([Q,P]);assert not (OLD/'controller.exit').exists()
legacy=js(E/'HANDOFF_COMPLETE.json')['legacy_completed_cases']
with (E/'FINAL_CPU_REQUEST.json').open('x') as f:json.dump(dict(observed=time.time(),status='RUNNING_NO_REPEAT'),f)
out=E/('final_cpu_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();results={}
for case in CASES:
    owner=OLD if case in legacy else E
    assert (owner/(case+'_evaluate.exit')).read_text().strip()=='0'
    en=env(case);en.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    p=subprocess.run([PY,'-B',str(Q/'final_case_cpu.py'),case],cwd=Q,env=en,capture_output=True,text=True)
    write(out/(case+'_ATTEMPT.json'),dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,(case,p.stderr[-2000:])
    r=json.loads(p.stdout.strip().splitlines()[-1]);assert r['status']=='PASS_CPU_CASE_RECONSTRUCTION';results[case]=r
verify();assert not processes([Q/'cases',P])
write(out/'CPU_RESULTS.json',dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cases=results,optimizer_calls=0,environments=0))
buf=io.StringIO()
with contextlib.redirect_stdout(buf):exec(compile((E/'check_readonly.py').read_text(),str(E/'check_readonly.py'),'exec'),{'__name__':'readonly_final'})
s=json.loads(buf.getvalue());s.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cpu=results,remote_cpu_directory=str(out));print(json.dumps(s))
