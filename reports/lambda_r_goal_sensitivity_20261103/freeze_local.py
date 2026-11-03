"""Bind successful readonly preflight and freeze reviewed text once."""
from pathlib import Path
import sys,json,hashlib,ast
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);S=R.parent/'20261029_support_gate_full_s_goal'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not (R/'DEPLOY_TEXT_BUNDLE.json').exists()
validation=js(R/'ENGINEERING_VALIDATION.json');assert validation['status']=='PASS_GOAL_FULL_PATH_BOTH_LAMBDAS' and validation['optimizer_started']==validation['optimizer_completed']==5704 and len(validation['stages'])==8 and validation['environments']==0
assert all(sha(R/n)==h for n,h in validation['reviewed_production_source_sha256'].items())
assert js(R/'OBJECT_COMPLETION_BINDING.json')['status']=='OBJECT_ACCEPTED_PUBLISHED_HOOK_CLEANED'
pre=js(D/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY' and pre['optimizer_calls']==pre['environments']==0
assert not pre['processes'] and not pre['gpu'] and all(x['headroom']>=0 for x in pre['capacity'].values())
cases=[g+'_'+str(seed) for g in ['L005','L020'] for seed in (41,42,43)]
bundle={};review=js(R/'SOURCE_REVIEW.json')
for case in cases:
 seed=int(case.split('_')[1]);p=js(R/case/'PARENT_BINDING.json');ref=pre['bindings']['GOAL_'+str(seed)]
 assert hashlib.sha256((S/('seed%d'%seed)/'PARENT_BINDING.json').read_text(encoding='utf8').encode()).hexdigest()==ref['parent_binding_sha256']
 assert p['output']==ref['output'] and p['source']==ref['source'] and p['stages']==ref['stages']
 assert js(R/case/'S_REFERENCE.json')==ref['accepted_S_reference']
 for n,x in review['files'].items():
  if n.startswith(case+'/'):assert hashlib.sha256((R/n).read_text(encoding='utf8').encode()).hexdigest()==x['new_sha256']
 for f in (R/case).iterdir():
  if f.is_file():
   assert f.suffix in ['.py','.json']
   text=f.read_text(encoding='utf8')
   if f.suffix=='.py':ast.parse(text,feature_version=(3,8))
   bundle[case+'/'+f.name]=text
 bundle[case+'/PROTOCOL.md']=(R/'PROTOCOL.md').read_text(encoding='utf8')
for n in ['controller_parallel.py','train_wrapper.py','environment_locked.py','ENGINEERING_VALIDATION.json','OBJECT_COMPLETION_BINDING.json','PROTOCOL.md']:bundle[n]=(R/n).read_text(encoding='utf8')
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(D/'EVIDENCE.json'),previous=pre['previous'],capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
dump(R/'PREFLIGHT_LOCAL_AUDIT.json',dict(status='PASS_PREDEPLOY_READONLY',sha256=sha(D/'EVIDENCE.json'),observed=pre['observed'],capacity=pre['capacity'],verified_unique_files=pre['verified_unique_files'],optimizer_calls=0,environments=0))
dump(R/'UPLOAD_REVIEW.json',dict(status='FROZEN_TEXT_ONLY_NOT_GPU_ACCEPTANCE',files={n:dict(bytes=len(t.encode()),sha256=hashlib.sha256(t.encode()).hexdigest()) for n,t in bundle.items()},private_payloads_included=False))
dump(R/'DEPLOY_TEXT_BUNDLE.json',bundle)
print(json.dumps(dict(status='FROZEN_NOT_DISPATCHED',files=len(bundle),epochs=2400,accepted_updates=456600,physical_calls=853200)))
