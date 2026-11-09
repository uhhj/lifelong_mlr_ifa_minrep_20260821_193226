"""Unique finite OBJECT then GOAL deployment; existing S remains read only."""
import os,json,hashlib,subprocess,time
from pathlib import Path
BASE=Path('/data/mlr_ifa/experiment_workspaces');NAME='S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1';Q=BASE/NAME
PY='/data/mlr_ifa/env/bin/python';CASES=['Q1_41','Q1_42','Q1_43']
roots={s:BASE/('S_Q1_'+s+'41_42_43_20261109_rerun_v1') for s in ['OBJECT','GOAL']}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
js=lambda p:json.loads(Path(p).read_text())
assert not Q.exists()
for r in roots.values():assert not r.exists() and not (Path('/data/mlr_ifa/experiments')/r.name).exists()
pre=json.loads(BUNDLE['PREDEPLOY.json']);assert 0<=time.time()-pre['observed']<3600
inputs=json.loads(BUNDLE['INPUT_BINDING.json'])
assert hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()==pre['input_manifest_sha256']
def readiness():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()=='NVIDIA GeForce RTX 4080 SUPER'
 for p in Path('/proc').glob('[0-9]*'):
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:continue
  if int(p.name)!=os.getpid():assert 'mlr_ifa' not in cmd and 'torch.distributed' not in cmd,('Existing process',p.name)
 for mount,bound,reserve in [('/root',128*1024**2,1024**3),('/data',6*1024**3,2*1024**3)]:
  s=os.statvfs(mount);assert s.f_bavail*s.f_frsize-bound>=reserve
 assert not Path('/data/mlr_ifa/env/lib/python3.8/site-packages/zzz_mlr_lambda_live_collection_20261102.pth').exists()
readiness()
for p,h in inputs.items():assert sha(p)==h,('Input hash mismatch',p)
readiness()
for n,s in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ['.py','.md','.json']
 if n.endswith('.py'):compile(s,n,'exec')
Q.mkdir()
for r in roots.values():r.mkdir()
allfiles={};suitefiles={s:{} for s in roots}
for n,s in BUNDLE.items():
 parts=Path(n).parts;suite=parts[0] if parts[0] in roots else None
 p=(roots[suite]/Path(*parts[1:])) if suite else Q/n
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
 allfiles[str(p)]=sha(p)
 if suite:suitefiles[suite][str(p)]=sha(p)
result={}
for suite,q in roots.items():
 acc,phys,gates=(105500,195300,9100) if suite=='OBJECT' else (76100,142200,6700)
 children={}
 for case in CASES:
  r=q/case;parent=js(r/'PARENT_BINDING.json');assert sum(x['batches']*100 for x in parent['stages'].values())==acc
  c=dict(files={p:h for p,h in suitefiles[suite].items() if str(r)+'/' in p},parent_source=parent['source_sha256'],source_root=parent['source'],official_initialization=parent['official_initialization'],seed=parent['seed'],suite=suite,method='S_FIXED_Q_ONE',lambda_R=.1,support_calibration='DISABLED_FIXED_ONE',budget=dict(epochs=400,accepted_updates=acc,physical_optimizer_calls=phys,C_candidate_calls=phys-acc,admission_optimizer_calls=91,admission_extra_autograd=41,collection_episodes=40,formal_episodes=200),created=time.time())
  (r/'CONTRACT.json').write_text(json.dumps(c,indent=2));children[case]=sha(r/'CONTRACT.json');suitefiles[suite][str(r/'CONTRACT.json')]=children[case];allfiles[str(r/'CONTRACT.json')]=children[case]
  code="import runtime,train,collect,evaluate,gate_admission; from support_memory import weights; m={6:dict(support=.2,rows=[None]),7:dict(support=.6,rows=[None]),8:dict(support=0.,rows=[])}; assert weights(m,'Q')=={6:1.,7:1.,8:0.}; assert weights(m,'G')==weights(m,'Q'); assert runtime.CALLS=={'production':0,'admission':0}; print('PASS_IMPORT_FIXED_Q_NO_UPDATES_NO_ENVIRONMENT')"
  p=subprocess.run([PY,'-B','-c',code],cwd=r,capture_output=True,text=True)
  (r/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,optimizer_calls=0,environments=0)))
  assert p.returncode==0,(suite,case,p.stdout,p.stderr)
 (q/'CONTRACT.json').write_text(json.dumps(dict(files=suitefiles[suite],children=children,created=time.time(),budget=dict(epochs=1200,accepted_updates=3*acc,physical_optimizer_calls=3*phys,admission_optimizer_calls=273,admission_extra_autograd=123,training_episodes=120,formal_episodes=600)),indent=2))
 allfiles[str(q/'CONTRACT.json')]=sha(q/'CONTRACT.json')
 result[suite]=dict(root=str(q),contract=sha(q/'CONTRACT.json'),children=children,frozen_files=len(suitefiles[suite]))
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=allfiles,suites=result,created=time.time(),input_count=len(inputs),input_manifest_sha256=pre['input_manifest_sha256'],budget=dict(epochs=2400,accepted_updates=544800,physical_optimizer_calls=1012500,training_episodes=240,formal_episodes=1200)),indent=2))
with (Q/'controller.log').open('x') as f:
 p=subprocess.Popen([PY,'-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_PENDING_OWN_GPU_ADMISSION',root=str(Q),controller=p.pid,contract=sha(Q/'CONTRACT.json'),suites=result,frozen_files=len(allfiles),verified_inputs=len(inputs),observed=time.time())))
