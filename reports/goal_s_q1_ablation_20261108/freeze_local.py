"""Freeze reviewed new sources once, only after fresh actual preflight."""
from pathlib import Path
import json,hashlib,ast,sys,time
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);CASES=['Q1_41','Q1_42','Q1_43']
read=lambda p:p.read_text(encoding='utf-8-sig');js=lambda p:json.loads(read(p));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'DEPLOY_TEXT_BUNDLE.json').exists()
pre=js(D/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY' and time.time()-pre['observed']<3600
assert pre['optimizer_calls']==pre['environments']==0 and not pre['processes'] and not pre['gpu_processes']['stdout'].strip()
assert all(x['headroom']>=0 for x in pre['capacity'].values())
review=js(R/'PREDEPLOY_REVIEW.json');assert review['status']=='PASS_STATIC_REVIEW_NOT_SCIENTIFIC_ACCEPTANCE'
assert all(sha(R/n)==h for n,h in review['files'].items())
bundle={}
for case in CASES:
 p=js(R/case/'PARENT_BINDING.json');assert p['lambda_R']==.1 and p['suite']=='GOAL' and p['support_calibration']=='DISABLED_FIXED_ONE'
 expected=dict(p['files']);expected.update({p['source']+'/'+n:h for n,h in p['source_sha256'].items()})
 expected.update({v['path']:v['sha256'] for v in js(R/case/'S_REFERENCE.json').values()})
 assert all(pre['verified_inputs'][n]==h for n,h in expected.items())
 for f in (R/case).iterdir():
  if not f.is_file():continue
  assert f.suffix in ['.py','.json'];s=read(f)
  if f.suffix=='.py':ast.parse(s,feature_version=(3,8))
  bundle[case+'/'+f.name]=s
 bundle[case+'/PROTOCOL.md']=read(R/'PROTOCOL.md')
for name in ['controller_parallel.py','train_wrapper.py','environment_locked.py','ENGINEERING_REUSE.json','PROTOCOL.md']:
 bundle[name]=read(R/name)
bundle['PREDEPLOY.json']=json.dumps(dict(pre,evidence_sha256=sha(D/'EVIDENCE.json')),indent=2)+'\n'
for name,s in bundle.items():
 if name.endswith('.py'):ast.parse(s,feature_version=(3,8))
manifest={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()}
(R/'UPLOAD_REVIEW.json').write_text(json.dumps(dict(status='FROZEN_TEXT_ONLY_NOT_SCIENTIFIC_ACCEPTANCE',files=manifest,private_payloads_included=False),indent=2),encoding='utf8')
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(status='FROZEN_NOT_DEPLOYED',files=len(bundle),epochs=1200,production_optimizer_calls=426600)))
