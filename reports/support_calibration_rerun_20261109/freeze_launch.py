"""Freeze the authorized six-run text package; no remote operations."""
from pathlib import Path
import ast,hashlib,json,time
R=Path(__file__).resolve().parent
read=lambda p:p.read_text(encoding='utf-8-sig')
js=lambda p:json.loads(read(p))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'DEPLOY_TEXT_BUNDLE.json').exists()
review=js(R/'CONFIGURATION_REVIEW.json')
assert all(sha(R/n)==v['new_sha256'] for n,v in review['files'].items())
bundle={'controller.py':read(R/'controller.py'),'PROTOCOL.md':read(R/'PROTOCOL.md')}
inputs={}
for suite in ['OBJECT','GOAL']:
 d=R/suite
 env=d/'environment_locked.py'
 txt=read(env).replace('S_Q1_'+suite+'41_42_43_20261109_rerun_v1_environment.lock','S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1_environment.lock')
 env.write_text(txt,encoding='utf8',newline='\n')
 (d/'PROTOCOL.md').write_text(read(R/'PROTOCOL.md'),encoding='utf8',newline='\n')
 baseline=js(d/'S_ACCEPTANCE.json')['sources'][suite]
 for name,key in [('EVIDENCE.json','evidence_sha256'),('LOCAL_FINAL_AUDIT.json','audit_sha256')]:
  assert sha(Path(baseline['final_path'])/name)==baseline[key]
 for seed in [41,42,43]:
  case='Q1_'+str(seed);p=js(d/case/'PARENT_BINDING.json')
  assert p['suite']==suite and p['seed']==seed and p['lambda_R']==.1 and p['support_calibration']=='DISABLED_FIXED_ONE'
  expected=dict(p['files']);expected.update({p['source']+'/'+n:h for n,h in p['source_sha256'].items()})
  expected.update({v['path']:v['sha256'] for v in js(d/case/'S_REFERENCE.json').values()})
  for n,h in expected.items():
   assert n not in inputs or inputs[n]==h
   inputs[n]=h
  for f in (d/case).iterdir():
   assert f.is_file() and f.suffix in ['.py','.json']
   bundle[suite+'/'+case+'/'+f.name]=read(f)
  bundle[suite+'/'+case+'/PROTOCOL.md']=read(R/'PROTOCOL.md')
 for name in ['controller_parallel.py','train_wrapper.py','environment_locked.py','ENGINEERING_REUSE.json','PROTOCOL.md']:
  bundle[suite+'/'+name]=read(d/name)
pre=js(R/'capacity_readonly_20261109_153402/EVIDENCE.json')
assert pre['status']=='PASS_POST_CLEANUP_READONLY' and not any(pre['planned_roots_exist'].values())
assert pre['free_bytes']['/data']>=8*1024**3 and pre['free_bytes']['/']>=1152*1024**2
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],capacity=pre['free_bytes'],input_count=len(inputs),input_manifest_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()))
bundle['INPUT_BINDING.json']=json.dumps(inputs,sort_keys=True)
for n,s in bundle.items():
 if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
manifest={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()}
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(dict(status='FROZEN_TEXT_ONLY_PENDING_ACTUAL_REMOTE_INPUT_VERIFICATION',files=manifest,allowed_additional_changes=['shared environment lock path','finite OBJECT then GOAL coordinator'],science_change_from_previous_q1=False,private_payloads_included=False),indent=2),encoding='utf8')
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(files=len(bundle),inputs=len(inputs),text_bytes=sum(v['bytes'] for v in manifest.values()),status='FROZEN_NOT_DEPLOYED')))
