"""One actual final CPU request after all new scientific jobs exit naturally."""
from pathlib import Path
import contextlib,io,json,os,runpy,subprocess,sys,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering02')
sys.path.insert(0,str(Q))
from reverse_common import *
assert (Q/'controller.exit').read_text().strip()=='0' and not processes()
verify(full=True)
with (Q/'FINAL_CPU_REQUEST.json').open('x') as f:json.dump(dict(status='RUNNING_NO_REPEAT',time=time.time(),pid=os.getpid()),f)
directory=Q/('final_cpu_'+time.strftime('%Y%m%d_%H%M%S'));directory.mkdir();results={}
for case in read(Q/'SPEC.json')['cases']:
    env=environment();env['CUDA_VISIBLE_DEVICES']=''
    p=subprocess.run([PY,'-B',str(Q/'final_cpu_case.py'),case],cwd=Q,env=env,capture_output=True,text=True)
    write(directory/(case+'_ATTEMPT.json'),dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0,('CPU wrapper failed; preserve attempt; no scientific rerun',case,str(directory),p.stderr[-2500:])
    result=json.loads(p.stdout.strip().splitlines()[-1]);assert result['status']=='PASS_REVERSE_CPU_CASE'
    results[case]=result
assert len(results)==12 and sum(x['formal'] for x in results.values())==2400
assert sum(x['training_collection'] for x in results.values())==240
assert sum(x['accepted_updates'] for x in results.values())==1156800
assert sum(x['physical_optimizer']['production'] for x in results.values())==1636500
assert sum(x['physical_optimizer']['admission'] for x in results.values())==666
verify(full=True);assert not processes()
write(directory/'CPU_RESULTS.json',dict(status='PASS_REMOTE_REVERSE_CPU_PENDING_LOCAL_ACCEPTANCE',cases=results,new_optimizer=0,new_environments=0))
buf=io.StringIO()
with contextlib.redirect_stdout(buf):runpy.run_path(str(Q/'check_readonly.py'),run_name='__main__')
snapshot=json.loads(buf.getvalue().strip().splitlines()[-1])
snapshot.update(status='PASS_REMOTE_REVERSE_CPU_PENDING_LOCAL_ACCEPTANCE',cpu=results,remote_cpu_directory=str(directory))
print(json.dumps(snapshot))
