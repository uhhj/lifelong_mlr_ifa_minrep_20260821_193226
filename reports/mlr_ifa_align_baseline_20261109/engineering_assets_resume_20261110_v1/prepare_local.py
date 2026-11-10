"""Prepare once, binding actual zero-policy-rollout failure and canonical assets."""
from pathlib import Path
import json,hashlib,ast
D=Path(__file__).resolve().parent
assert not (D/'DEPLOY_TEXT_BUNDLE.json').exists()
p=D/'preflight_readonly_20261110_151738/EVIDENCE.json';e=json.loads(p.read_text(encoding='utf8'))
assert e['status']=='PASS_READONLY_ASSETS_AND_PRE_ROLLOUT_FAILURE' and e['worker_file_not_found_count']==10
files={n:(D/n).read_text(encoding='utf8') for n in ['resume_runtime.py','controller_resume.py','evaluate_guarded.py','check_readonly.py','deploy_remote.py']}
files['ASSETS.json']=json.dumps(dict(root=e['asset_root'],manifest=e['asset_manifest']),indent=2)+'\n'
files['PREFLIGHT_BINDING.json']=json.dumps(dict(evidence_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),failed_evaluation_hashes=e['failed_evaluation_hashes'],old_log_sha256=e['log_sha256'],worker_file_not_found_count=10,attempt_events=e['attempt_events']),indent=2)+'\n'
for n,t in files.items():
 if n.endswith('.py'):ast.parse(t)
 else:(D/n).write_text(t,encoding='utf8')
c=dict(status='FROZEN_ENGINEERING_ONLY',science_contract='ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f',files={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()},changes=['Link six missing assets directories to existing canonical immutable assets','Preserve original failed evaluation directory and exact RNG','Guard exact reconstructed pre-cell RNG before any environment','Resume OBJECT evaluations then unchanged GOAL training/evaluations'],no_training_repeated=True,no_original_science_source_changes=True)
files['CONTRACT.json']=json.dumps(c,indent=2)+'\n';(D/'CONTRACT.json').write_text(files['CONTRACT.json'],encoding='utf8')
(D/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(files),encoding='utf8')
(D/'REVIEW.json').write_text(json.dumps(dict(status='STATIC_REVIEW_PENDING_ACTUAL_RNG_AND_ENVIRONMENT',contract_sha256=hashlib.sha256(files['CONTRACT.json'].encode()).hexdigest(),preflight=e['status'],source_files=len(c['files']),optimizer_diagnostic_added=0,environment_validation_added=0),indent=2)+'\n',encoding='utf8')
print(hashlib.sha256(files['CONTRACT.json'].encode()).hexdigest())
