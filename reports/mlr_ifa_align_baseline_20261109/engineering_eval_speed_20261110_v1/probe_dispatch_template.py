from pathlib import Path
import sys,os,json,time,subprocess,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(E));from resume_runtime import verify,env,P
verify()
D=Path('/data/mlr_ifa/diagnostics/align_eval_offline_cache_20261110_v1');assert not D.exists(),'Never repeat this probe'
r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert r.returncode==0 and int(r.stdout.strip())>4200,r.stdout
D.mkdir();p=D/'probe.py';p.write_text(SOURCE,encoding='utf8');environment=env('OBJECT_41');environment['PYTHONDONTWRITEBYTECODE']='1'
log=D/'probe.log'
with log.open('x') as f:
 r=subprocess.run(['/data/mlr_ifa/env/bin/python','-B',str(p)],cwd=str(Q/'cases/OBJECT_41/source'),env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=240)
text=log.read_text();result=[s[len('RESULT_JSON='):] for s in text.splitlines() if s.startswith('RESULT_JSON=')]
verify()
print(json.dumps(dict(status='OFFLINE_PROBE_FINISHED',exit=r.returncode,result=json.loads(result[-1]) if result else None,log_tail=text[-5000:] if not result else None,source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),remote_dir=str(D),new_optimizer=0,new_environment=0,production_writes=0)))
