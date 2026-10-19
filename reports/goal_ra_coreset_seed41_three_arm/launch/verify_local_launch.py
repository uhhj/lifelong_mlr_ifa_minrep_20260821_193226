"""Read-only verification of deployment hashes, actual configs and GPU evidence."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=sorted(R.glob('progress_*'))[-1];S=js(D/'SNAPSHOT.json');B=D/'baseline';C=js(B/'CONTRACT.json')
assert js(R/'DEPLOY_COMMAND.json')['exit_code']==0
for n,h in js(R/'CODE_MANIFEST.json').items():assert sha(R/n)==h,n
for arm,proof in js(R/'SOURCE_DIFF.json').items():
 for n,h in {**proof['unchanged'],**proof['changed']}.items():assert sha(R/arm/'source'/n)==h
 assert proof['scientific_core_unchanged']
assert C['seed']==41 and C['modal_sha256']==sha(B/'MODAL_CONTRACT.json')
assert C['orchestration_sha256']==sha(R/'ORCHESTRATION_SHA256.json')
configs={}
for name in ['export','random','coreset','eval']:
 c=js(B/f'resolved_{name}.json');assert c['seed']==41 and c['benchmark_name']=='LIBERO_GOAL' and c['gamma']==.7
 assert all(c['policy'][n] is True for n in ['use_vision','use_language','use_state'])
 assert c['train']['batch_size']==(1 if name=='export' else 10) and c['train']['n_epochs']==100 and c['eval']['n_eval']==20
 configs[name]=sha(B/f'resolved_{name}.json')
g=js(B/'GPU_PREFLIGHT.json');assert g['status']=='PASS' and g['contract_sha256']==sha(B/'CONTRACT.json')
assert g['tests']['isolated_optimizer_calls']==4 and g['formal_rollouts']==0 and g['environment_interface_checks']==1
assert g['tests']['state_token_exact'] and g['tests']['gmm_parameters_exact'] and g['tests']['optimizer_scheduler_rng_recovery_exact']
for n in ['baseline_software','baseline_configure','baseline_suite_audit','baseline_preflight']:assert S['phase_exits'][n+'.exit']=='0'
assert 'Ran 18 tests' in (D/'queue/baseline_software.log').read_text(encoding='utf-8')
assert S['failure'] is None and S['queue_exit'] is None and S['wrapper_exit'] is None
out=dict(status='PASS_LAUNCH_ONLY_PARTIAL',time=datetime.datetime.now().astimezone().isoformat(),snapshot=str(D.relative_to(R)),source_and_orchestration_hashes_verified=True,configs=configs,software_tests=18,gpu_preflight=g,RA_preflight='PENDING_AFTER_BASELINE',formal_status=S['arms'],expected=dict(epochs=1200,optimizer_steps=228300,checkpoints=12,records=600,vector_attempts=60))
(R/'LOCAL_LAUNCH_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
(R/'RUN_STATUS.json').write_text(json.dumps(dict(status='PARTIAL',phase=S['arms']['random']['status']['status'],time=out['time'],seed=41,arms=['random','coreset','ra_coreset'],automation='goal-seed41',snapshot=out['snapshot']),indent=2)+'\n')
print(json.dumps(dict(status=out['status'],software_tests=18,gpu='PASS',phase=S['arms']['random']['status']['status'])))
