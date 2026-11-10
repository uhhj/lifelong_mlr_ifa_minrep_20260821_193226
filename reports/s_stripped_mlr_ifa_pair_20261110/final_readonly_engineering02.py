"""External CPU-only wrapper compatibility; frozen science and old wrapper retained."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
Q=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
sys.path.insert(0,str(Q))
from runtime import *
assert (Q/'controller.exit').read_text().strip()=='0' and not processes()
verify(full=True)
with (Q/'FINAL_CPU_REQUEST_engineering02.json').open('x') as f:json.dump(dict(status='RUNNING_NO_REPEAT',time=time.time(),pid=os.getpid()),f)
out=Q/('final_cpu_engineering02_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();results={}
old=(Q/'final_case_cpu.py').read_text()
needle='from libero.lifelong.models.brief import BriefPolicy'
assert old.count(needle)==1 and old.count("CUDA_VISIBLE_DEVICES=''")==1
adapted=old.replace("CUDA_VISIBLE_DEVICES=''","CUDA_VISIBLE_DEVICES='0'",1)
# The inherited LIBERO metric imports EGL from a parseable device index.
# CPU verification reads saved CUDA RNG bytes without executing a CUDA RNG API.
adapted=adapted.replace(needle,needle+"\n\ndef rng():\n    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),cuda=[])\n",1)
write(out/'WRAPPER_CHANGE.json',dict(status='EXTERNAL_CPU_WRAPPER_ONLY',original_script_sha256=hashlib.sha256(old.encode()).hexdigest(),executed_script_sha256=hashlib.sha256(adapted.encode()).hexdigest(),changes=['parseable inherited EGL import device','CPU replay RNG inspection excludes CUDA execution; saved CUDA state bytes still verified'],frozen_sources_modified=False,training_or_environment_rerun=False))
code="namespace={'__name__':'__main__','__file__':%r};exec(compile(%r,%r,'exec'),namespace)"%(str(Q/'final_case_cpu.py'),adapted,str(Q/'final_case_cpu.py'))
for case in CASES:
    en=env(case);en['CUDA_VISIBLE_DEVICES']='0'
    p=subprocess.run([PY,'-B','-c',code,case],cwd=Q,env=en,capture_output=True,text=True)
    write(out/(case+'_ATTEMPT.json'),dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0,('CPU wrapper failed; preserve attempt; do not rerun science',case,str(out),p.stderr[-2500:])
    result=json.loads(p.stdout.strip().splitlines()[-1]);assert result['status']=='PASS_CPU_CASE_RECONSTRUCTION';results[case]=result
verify(full=True);assert not processes()
write(out/'CPU_RESULTS.json',dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cases=results,new_optimizer=0,new_environments=0))
s=snapshot();s.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cpu=results,remote_cpu_directory=str(out),external_cpu_wrapper='engineering02',wrapper_change=js(out/'WRAPPER_CHANGE.json'));print(json.dumps(s))
