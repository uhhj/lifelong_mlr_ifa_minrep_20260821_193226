"""Build an isolated, single-use GOAL validation request after OBJECT closure."""
from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;R=D.parent
read=lambda p:p.read_text(encoding='utf8');sha=lambda p:hashlib.sha256(read(p).encode()).hexdigest()
closure=json.loads(read(R/'OBJECT_COMPLETION_BINDING.json'))
assert closure['status']=='OBJECT_ACCEPTED_PUBLISHED_HOOK_CLEANED'
reviewed={p.relative_to(R).as_posix():sha(p) for p in R.glob('L*_*/*') if p.is_file()}
reviewed.update({n:sha(R/n) for n in ['train_wrapper.py','environment_locked.py','controller_parallel.py']})
source={n:read(R/n) for n in reviewed}
validation=read(D/'validate.py')
validation=validation.replace("assert not (D/'VALIDATION_PROGRESS.json').exists(),'single use'","assert not (D/'VALIDATION_PROGRESS.json').exists(),'single use'\n assert rt.js(D/'LOCK_VALIDATION.json')['status']=='PASS_ZERO_ENV_EXACT_SUBPROCESS_MUTEX'\n ledger['reviewed_production_source_sha256']=rt.js(D/'INPUT_BINDING.json')['reviewed_production_source_sha256']\n ledger['lock_validation_sha256']=rt.sha(D/'LOCK_VALIDATION.json')")
runner="from pathlib import Path\nimport subprocess,sys\nD=Path(__file__).resolve().parent\ncode=1\ntry:\n code=subprocess.call([sys.executable,'-B',str(D/'validate_environment_lock.py')],cwd=D)\n if code==0:code=subprocess.call([sys.executable,'-B',str(D/'validate.py')],cwd=D)\nfinally:\n with (D/'validation.exit').open('x') as f:f.write(str(code))\nsys.exit(code)\n"
prefix='SOURCES='+repr(source)+'\nVALIDATION_SOURCE='+repr(validation)+'\nLOCK_SOURCE='+repr(read(D/'validate_environment_lock.py'))+'\nRUNNER_SOURCE='+repr(runner)+'\nBINDING='+repr(dict(object_closure=closure,reviewed_production_source_sha256=reviewed,source_digest_encoding='UTF8_LF_TEXT_MATCHES_FROZEN_BUNDLE'))+'\n'
body='''import os,sys,json,time,hashlib,subprocess
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_r_goal_engineering_20261103_v1')
P=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_GOAL41_42_43_20261103_v1')
O=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
assert not D.exists() and not P.exists(),'Single diagnostic dispatch, no deployed GOAL root allowed'
assert (O/'engineering_cross_lambda_20261102_v1/controller.exit').read_text().strip()=='0'
assert json.loads((O/'OWNED_HOOK_CLEANUP.json').read_text())['status']=='PASS_OWNED_HOOK_CLEANUP_AFTER_ACCEPTANCE'
assert not Path('/data/mlr_ifa/env/lib/python3.8/site-packages/zzz_mlr_lambda_live_collection_20261102.pth').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert BINDING['object_closure']['status']=='OBJECT_ACCEPTED_PUBLISHED_HOOK_CLEANED'
sha=lambda p:hashlib.sha256(read(p).encode()).hexdigest()
for case in [g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]:
 parent=json.loads(SOURCES[case+'/PARENT_BINDING.json'])
 assert all(sha(Path(p))==h for p,h in parent['files'].items())
 assert all(sha(Path(parent['source'])/n)==h for n,h in parent['source_sha256'].items())
for n,s in SOURCES.items():
 assert hashlib.sha256(s.encode()).hexdigest()==BINDING['reviewed_production_source_sha256'][n]
 if n.endswith('.py'):compile(s,n,'exec')
assert not (Path('/data/mlr_ifa/experiments')/P.name).exists()
D.mkdir();Q=D/'source';Q.mkdir();files={}
for n,s in SOURCES.items():
 p=Q/n;p.parent.mkdir(exist_ok=True);p.write_text(s);files[str(p)]=sha(p)
for case in [g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]:
 child=Q/case;p=json.loads((child/'PARENT_BINDING.json').read_text())
 (child/'CONTRACT.json').write_text(json.dumps(dict(files={n:h for n,h in files.items() if n.startswith(str(child)+'/')},parent_source=p['source_sha256'],source_root=p['source'],diagnostic_only=True),indent=2))
 files[str(child/'CONTRACT.json')]=sha(child/'CONTRACT.json')
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,diagnostic_only=True),indent=2))
for name,s in [('validate.py',VALIDATION_SOURCE),('validate_environment_lock.py',LOCK_SOURCE),('runner.py',RUNNER_SOURCE)]:compile(s,name,'exec');(D/name).write_text(s)
(D/'INPUT_BINDING.json').write_text(json.dumps(BINDING,indent=2))
with (D/'validation.log').open('x') as f:
 proc=subprocess.Popen([sys.executable,'-B',str(D/'runner.py')],cwd=D,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
out=dict(status='GOAL_DIAGNOSTIC_ONLY_DISPATCHED',directory=str(D),pid=proc.pid,observed=time.time(),validation_source_sha256=sha(D/'validate.py'),runner_sha256=sha(D/'runner.py'),source_contract_sha256=sha(Q/'CONTRACT.json'),max_optimizer_calls=5704,max_extra_autograd=36,environments=0,production_epochs=0,production_root_created=False)
(D/'DISPATCH.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
code=prefix+body;ast.parse(code,feature_version=(3,8));(D/'dispatch_validation_v2.py').write_text(code,encoding='utf8')
(D/'LOCAL_DISPATCH_REVIEW_v2.json').write_text(json.dumps(dict(status='REVIEWED_SINGLE_DIAGNOSTIC_NOT_PRODUCTION',dispatch_sha256=hashlib.sha256(code.encode()).hexdigest(),source_files=reviewed,max_optimizer_calls=5704,max_extra_autograd=36,environments=0,production_epochs=0,object_closure=closure),indent=2)+'\n',encoding='utf8')
