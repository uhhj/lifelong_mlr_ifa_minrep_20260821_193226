"""Actual CPU acceptance only after natural success of all twelve runs."""
from pathlib import Path
import sys,json,subprocess,os,time
Q=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
sys.path.insert(0,str(Q))
from runtime import *
assert (Q/'controller.exit').read_text().strip()=='0' and not processes(),'Controller must naturally finish first'
verify(full=True)
with (Q/'FINAL_CPU_REQUEST.json').open('x') as f:json.dump(dict(status='RUNNING_NO_REPEAT',time=time.time(),pid=os.getpid()),f)
out=Q/('final_cpu_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();results={}
for case in CASES:
 e=env(case);e['CUDA_VISIBLE_DEVICES']=''
 p=subprocess.run([PY,'-B',str(Q/'final_case_cpu.py'),case],cwd=Q,env=e,capture_output=True,text=True)
 write(out/(case+'_ATTEMPT.json'),dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
 assert p.returncode==0,('CPU acceptance failed; preserve attempt, no scientific rerun',case,str(out),p.stderr[-2000:])
 result=json.loads(p.stdout.strip().splitlines()[-1]);assert result['status']=='PASS_CPU_CASE_RECONSTRUCTION',result.get('status');results[case]=result
verify(full=True);assert not processes()
write(out/'CPU_RESULTS.json',dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cases=results,new_optimizer=0,new_environments=0))
s=snapshot();s.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cpu=results,remote_cpu_directory=str(out));print(json.dumps(s))
