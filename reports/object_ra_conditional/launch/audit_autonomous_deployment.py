"""Read-only local deployment verification; not experimental-result acceptance."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;E=R/'deployment_20261019';N=R/'server_auto'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=js(E/'DEPLOYMENT_DOWNLOAD_MANIFEST.json')
for name,v in manifest.items():
 assert sha(E/name)==v['sha256'] and (E/name).stat().st_size==v['bytes'],name
c=js(N/'AUTONOMOUS_CONTRACT.json')
for name,h in c['files'].items():assert sha(N/name)==h,name
assert sha(N/'AUTONOMOUS_CONTRACT.json')==js(E/'AUTONOMOUS_LAUNCH.json')['contract_sha256']==js(E/'DEPLOYMENT.json')['contract_sha256']
p=js(E/'stage1/41/PROTOCOL_EQUIVALENCE.json');a=js(E/'stage1/41/TASK6_POLICY_ACCEPTANCE.json')
assert p['status']=='PASS_NO_PROTOCOL_DEVIATION_FOUND' and p['initial_model_and_all_rng_equal']
assert all(p['training_functions_AST_same'].values()) and p['dataset_order_and_payload_equal']
assert len(p['epoch_order_sha256'])==100 and all(v['current']==v['parent'] for v in p['critical_source_hashes'].values())
assert a['protocol_audit_sha256']==sha(E/'stage1/41/PROTOCOL_EQUIVALENCE.json')
assert a['status']=='DISCLOSED_CROSS_HARDWARE_DIFFERENCE' and a['initial_model_rng_equal']
assert p['resume_authority']['committed_steps']==15543 and p['resume_authority']['repeated_uncommitted_epoch99_steps']==157
for name in ['target_tests','admission_inventory','controller_paths']:assert (E/(name+'.exit')).read_text().strip()=='0'
paths=js(E/'controller_paths.log');assert paths['synthetic'] and len(paths['cases'])==4 and all(x['status']=='PASS' for x in paths['cases'])
parents=js(E/'PREDEPLOY_RESOURCES.json');assert parents['status']=='PASS_READONLY_NOT_GPU_ADMISSION' and len(parents['parents'])==7
snap=js(E/'DEPLOYMENT_PROGRESS.json');assert all(v['alive'] for v in snap['processes'].values())
assert snap['seeds']['41']['epochs']==239 and snap['seeds']['41']['logical_steps']==50451
out=dict(status='PASS_DEPLOYMENT_ONLY_EXPERIMENT_PARTIAL',time=datetime.datetime.now().astimezone().isoformat(),observed_at_unix=snap['time'],metadata_files=len(manifest),frozen_source_files=len(c['files']),protocol_deviation_found=False,causal_hardware_explanation_proven=False,source_hashes='PASS',initial_model_all_rng='PASS',reconstructed_order_100_epochs='PASS',historical_actual_batch_trace_available=False,all_three_branch_paths_and_engineering_stop='PASS_SYNTHETIC',target_tests=7,existing_goal_parents=7,future_zrv2_gpu_admission='PENDING_ONLY_IF_BRANCH_SELECTED',training_status=snap['seeds'],controller_pid=snap['processes']['controller.started']['pid'],formal_results_complete=False)
(R/'AUTONOMOUS_LOCAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
