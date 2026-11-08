"""Single authorized Q1 deployment from locally reviewed text-only BUNDLE."""
import os,json,hashlib,subprocess,time
from pathlib import Path
NAME='S_Q1_GOAL41_42_43_20261108_v1';Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME;O=Path('/data/mlr_ifa/experiments')/NAME
CASES=['Q1_41','Q1_42','Q1_43'];PY='/data/mlr_ifa/env/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
assert not Q.exists() and not O.exists(),'No redeployment or output reuse'
pre=json.loads(BUNDLE['PREDEPLOY.json']);assert time.time()-pre['observed']<3600 and pre['status']=='PASS_PREDEPLOY_READONLY'
assert all(sha(p)==h for p,h in pre['verified_inputs'].items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()=='NVIDIA GeForce RTX 3090'
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 if int(p.name)!=os.getpid():assert 'mlr_ifa' not in cmd and 'torch.distributed' not in cmd,('Existing process',p.name)
for mount,bound,reserve in [('/root',128*1024**2,1024**3),('/data',4*1024**3,2*1024**3)]:
 s=os.statvfs(mount);assert s.f_bavail*s.f_frsize-bound>=reserve
assert not Path('/data/mlr_ifa/env/lib/python3.8/site-packages/zzz_mlr_lambda_live_collection_20261102.pth').exists()
for n,s in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ['.py','.md','.json']
 if n.endswith('.py'):compile(s,n,'exec')
Q.mkdir();files={}
for n,s in BUNDLE.items():
 p=Q/n;p.parent.mkdir(exist_ok=True);p.write_text(s);files[str(p)]=sha(p)
children={}
for case in CASES:
 r=Q/case;parent=js(r/'PARENT_BINDING.json');assert sum(x['batches']*100 for x in parent['stages'].values())==76100
 c=dict(files={p:h for p,h in files.items() if str(r)+'/' in p},parent_source=parent['source_sha256'],source_root=parent['source'],official_initialization=parent['official_initialization'],seed=parent['seed'],suite='GOAL',method='S_FIXED_Q_ONE',lambda_R=.1,support_calibration='DISABLED_FIXED_ONE',budget=dict(epochs=400,accepted_updates=76100,physical_optimizer_calls=142200,C_candidate_calls=66100,admission_optimizer_calls=91,admission_extra_autograd=41,collection_episodes=40,formal_episodes=200),created=time.time())
 (r/'CONTRACT.json').write_text(json.dumps(c,indent=2));children[case]=sha(r/'CONTRACT.json');files[str(r/'CONTRACT.json')]=children[case]
 code="import runtime,train,collect,evaluate,gate_admission; from support_memory import weights; m={6:dict(support=.2,rows=[None]),7:dict(support=.6,rows=[None]),8:dict(support=0.,rows=[])}; assert weights(m,'Q')=={6:1.,7:1.,8:0.}; assert weights(m,'G')==weights(m,'Q'); assert runtime.CALLS=={'production':0,'admission':0}; print('PASS_IMPORT_FIXED_Q_NO_UPDATES_NO_ENVIRONMENT')"
 p=subprocess.run([PY,'-B','-c',code],cwd=r,capture_output=True,text=True)
 (r/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,optimizer_calls=0,environments=0)))
 assert p.returncode==0,(case,p.stdout,p.stderr)
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,children=children,created=time.time(),budget=dict(epochs=1200,accepted_updates=228300,physical_optimizer_calls=426600,admission_optimizer_calls=273,admission_extra_autograd=123,training_episodes=120,formal_episodes=600)),indent=2))
with (Q/'controller.log').open('x') as f:
 p=subprocess.Popen([PY,'-B',str(Q/'controller_parallel.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_PENDING_OWN_GPU_ADMISSION',root=str(Q),controller=p.pid,contract=sha(Q/'CONTRACT.json'),children=children,frozen_files=len(files),hardware=pre['hardware'],observed=time.time())))
