from pathlib import Path
import sys,json,subprocess,hashlib,time
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
sys.path.insert(0,str(Q/'engineering_assets_resume_20261110_v2'));import resume_runtime as rt
rt.verify();D=Path('/data/mlr_ifa/diagnostics/align_accepted_cache_validation_20261110_v1');assert not D.exists(),'No duplicate validation'
D.mkdir()
for n,t in FILES.items():compile(t,n,'exec');(D/n).write_text(t,encoding='utf8')
rows={}
for case in ['OBJECT_43','GOAL_41']:
    with (D/(case+'.log')).open('x') as f:
        p=subprocess.run([rt.PY,'-B',str(D/'validate_cache.py')],cwd=Q/'cases'/case/'source',env=rt.env(case),stdout=f,stderr=subprocess.STDOUT,timeout=240)
    raw=(D/(case+'.log')).read_text();vals=[x[12:] for x in raw.splitlines() if x.startswith('RESULT_JSON=')]
    rows[case]=dict(exit=p.returncode,result=json.loads(vals[-1]) if vals else None,log_tail=raw[-4000:] if not vals else None)
    if p.returncode:break
rt.verify();result=dict(status='PASS_OFFLINE_ACCEPTED_CACHE_ADAPTATION' if len(rows)==2 and all(v['exit']==0 for v in rows.values()) else 'FAIL_OFFLINE_DO_NOT_DEPLOY',cases=rows,files={k:hashlib.sha256(v.encode()).hexdigest() for k,v in FILES.items()},observed=time.time(),new_optimizer=0,new_environment=0,production_writes=0)
rt.write(D/'VALIDATION.json',result);print(json.dumps(result))
