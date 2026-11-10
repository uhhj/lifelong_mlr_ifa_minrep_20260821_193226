from pathlib import Path
import sys,json,hashlib,subprocess,time,os
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v2';OLD=Q/'engineering_assets_resume_20261110_v1'
sys.path.insert(0,str(OLD));import resume_runtime as prior
P=prior.P
assert not E.exists(),'Existing attempt; do not repeat'
assert not prior.processes([Q,P]);prior.verify()
assert (OLD/'controller.exit').read_text().strip()=='1'
assert prior.sha(OLD/'CONTRACT.json')=='52cc16460a0f9d4a792ec27686245fe1a03c7bdd30fa2326844129aec12dae34'
assert prior.js(OLD/'RNG_BOUNDARY_AUDIT.json')['status']=='PASS_EXACT_PRE_CELL_RNG'
b=json.loads(BUNDLE['SECOND_FAILURE_BINDING.json']);failed=P/'OBJECT_41/random/evaluation'
assert {p.name:prior.sha(p) for p in failed.iterdir() if p.is_file()}==b['evaluation']
assert prior.sha(OLD/'OBJECT_41_evaluate.log')==b['log_sha256']
preserved=prior.js(OLD/'PRESERVED.json')['files'].copy()
for p in [OLD/'CONTRACT.json',OLD/'FAILURE.json',OLD/'controller.exit',OLD/'OBJECT_41_evaluate.log',OLD/'OBJECT_41_evaluate.exit',OLD/'EVENTS.jsonl',OLD/'RNG_BOUNDARY_AUDIT.json']:preserved[str(p)]=prior.sha(p)
for n,t in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts
 if n.endswith('.py'):compile(t,n,'exec')
c=json.loads(BUNDLE['CONTRACT.json']);assert all(hashlib.sha256(BUNDLE[n].encode()).hexdigest()==h for n,h in c['files'].items())
capacity=prior.original.capacity();E.mkdir()
for n,t in BUNDLE.items():
 with (E/n).open('x') as f:f.write(t)
os.rename(failed,E/'failed_evaluation_OBJECT41')
for n,h in b['evaluation'].items():preserved[str(E/'failed_evaluation_OBJECT41'/n)]=h
prior.write(E/'PRESERVED.json',dict(files=preserved,old_controller_exit=1,old_recovery_exit=1,attempts_retained=2))
# Validate through a fresh module namespace, not the imported old engineering helper.
scope={'__name__':'reviewed_resume_runtime'};exec(compile(BUNDLE['resume_runtime.py'],str(E/'resume_runtime.py'),'exec'),scope);scope['verify']()
with (E/'controller.log').open('x') as f:
 p=subprocess.Popen([prior.PY,'-B',str(E/'controller_resume.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
prior.write(E/'DEPLOYMENT.json',dict(status='STARTED_PENDING_ACTUAL_ROLLOUT',controller=p.pid,contract_sha256=prior.sha(E/'CONTRACT.json'),observed=time.time(),capacity=capacity,training_repeated=0,old_failures_preserved=True))
print(json.dumps(dict(status='STARTED_PENDING_ACTUAL_ROLLOUT',controller=p.pid,engineering_root=str(E),contract_sha256=prior.sha(E/'CONTRACT.json'))))
