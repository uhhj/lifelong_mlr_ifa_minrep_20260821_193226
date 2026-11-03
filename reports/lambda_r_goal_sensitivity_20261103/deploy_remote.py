"""One new six-S GOAL lambda-grid deployment. BUNDLE contains reviewed source/metadata only."""
import os,json,hashlib,subprocess,time
from pathlib import Path
NAME='S_LAMBDA_R_GOAL41_42_43_20261103_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME;O=Path('/data/mlr_ifa/experiments')/NAME
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
CASES=[s+'_'+str(k) for s in ['L005','L020'] for k in [41,42,43]]
assert not Q.exists() and not O.exists(),'No redeploy/output reuse'
pre=json.loads(BUNDLE['PREDEPLOY.json']);assert time.time()-pre['observed']<3600
validation=json.loads(BUNDLE['ENGINEERING_VALIDATION.json']);assert validation['status']=='PASS_GOAL_FULL_PATH_BOTH_LAMBDAS' and validation['optimizer_started']==validation['optimizer_completed']==5704
closure=json.loads(BUNDLE['OBJECT_COMPLETION_BINDING.json']);assert closure['status']=='OBJECT_ACCEPTED_PUBLISHED_HOOK_CLEANED'
object_root=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');assert (object_root/'engineering_cross_lambda_20261102_v1/controller.exit').read_text().strip()=='0'
assert not Path('/data/mlr_ifa/env/lib/python3.8/site-packages/zzz_mlr_lambda_live_collection_20261102.pth').exists()
assert not (object_root/'engineering_live_parallel_3090_20261102_v1/HOOK_ENABLED.json').exists()
old=Path('/data/mlr_ifa/experiment_workspaces/C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1')
assert (old/'engineering_suite_order_20261030_v1/controller.exit').read_text()=='0'
completed_q=Path('/data/mlr_ifa/experiment_workspaces/FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v2')
assert (completed_q/'controller.exit').read_text().strip()=='0'
assert js('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1/user_stop_20261028/RESULT.json')['status']=='USER_STOPPED_PARTIAL_PRESERVED_NOT_SCIENTIFIC_ACCEPTANCE'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 if int(p.name)!=os.getpid():assert all(x not in cmd for x in [str(Q),str(object_root),str(old),str(completed_q),*(v['root'] for v in pre['previous'].values()),'SUPPORT_GATE_FORKS_20261027_v1'])
for mount,bound,reserve,root in [('/root',128*1024**2,1024**3,False),('/data',12*1024**3,2*1024**3,True)]:
 st=os.statvfs(mount);assert (st.f_bfree if root else st.f_bavail)*st.f_frsize-bound>=reserve
for case in CASES:
 p=json.loads(BUNDLE[case+'/PARENT_BINDING.json']);assert all(sha(n)==h for n,h in p['files'].items())
 assert all(sha(Path(p['source'])/n)==h for n,h in p['source_sha256'].items())
 assert all(sha(v['path'])==v['sha256'] for v in json.loads(BUNDLE[case+'/S_REFERENCE.json']).values())
for n,s in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ['.py','.md','.json']
 if n.endswith('.py'):compile(s,n,'exec')
Q.mkdir();files={}
for n,s in BUNDLE.items():
 p=Q/n;p.parent.mkdir(exist_ok=True);p.write_text(s);files[str(p)]=sha(p)
children={}
for case in CASES:
 child=Q/case;parent=js(child/'PARENT_BINDING.json');total=sum(x['batches']*100 for x in parent['stages'].values())
 contract=dict(files={p:h for p,h in files.items() if str(child)+'/' in p},parent_source=parent['source_sha256'],source_root=parent['source'],official_initialization=parent['official_initialization'],seed=parent['seed'],suite=parent['suite'],method='S_support_gate',lambda_R=parent['lambda_R'],lambda_group=parent['lambda_group'],budget=dict(epochs=400,accepted_updates=total,physical_optimizer_calls=142200,C_candidate_calls=66100,admission_optimizer_calls=91,admission_extra_autograd=41,collection_episodes=40,formal_episodes=200),created=time.time())
 (child/'CONTRACT.json').write_text(json.dumps(contract,indent=2));children[case]=sha(child/'CONTRACT.json');files[str(child/'CONTRACT.json')]=children[case]
 proc=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-c','import runtime,memory,support_memory,fork_tools,gate_admission,updates,initialization,train,collect,evaluate; import synthetic_checks; print(synthetic_checks.run())'],cwd=child,capture_output=True,text=True)
 (child/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)))
 assert proc.returncode==0,(case,proc.stdout,proc.stderr)
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,children=children,created=time.time(),budget=dict(epochs=2400,accepted_updates=456600,physical_optimizer_calls=853200,C_candidate_calls=396600,admission_optimizer_calls=546,admission_extra_autograd=246,training_episodes=240,formal_episodes=1200)),indent=2))
with (Q/'controller.log').open('x') as f:
 proc=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller_parallel.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_PENDING_OWN_GPU_ADMISSION',root=str(Q),controller=proc.pid,contract=sha(Q/'CONTRACT.json'),children=children,frozen_files=len(files),hardware=pre['hardware'],observed=time.time())))
