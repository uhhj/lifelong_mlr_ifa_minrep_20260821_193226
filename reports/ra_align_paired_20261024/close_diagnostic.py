"""Bind completed local diagnostics to accepted initial-state hashes and costs."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'final_20261024_193008'; S=R/'snapshot_20261024_192943'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
a=js(D/'LOCAL_FINAL_AUDIT.json'); snap=js(S/'SNAPSHOT.json')
assert a['status']=='PASS_LOCAL_PAIRED_DIAGNOSTIC_AUDIT'
assert snap['source_unchanged'] and not snap['processes'] and not snap['gpu'].strip()
for name in ['worker.exit','supervisor.exit']:assert (D/name).read_text().strip()=='0'
out=js(D/'OUTCOMES.json'); events=[json.loads(x) for x in (D/'EVENTS.jsonl').read_text().splitlines()]
accepted=[]
base=R.parent/'20261022_align_object_extension_goal_conditional'
for seed in [41,42,43]:
 p=next(base.glob(f'final_GOAL_{seed}_*/current/ra_coreset/evaluation/trials_stage8_task8.jsonl'))
 row=json.loads(p.read_text().splitlines()[0])
 for o in out:
  if o['seed']==seed:assert o['initial_sha256']==row['initial_hashes'][o['initial_index']]
 for e in events:
  if e['kind']=='vector_start' and e['seed']==seed:assert e['initial_file_sha256']==row['initial_file_sha256']
 accepted.append({'seed':seed,'previously_accepted_trial_log':str(p),'sha256':sha(p),'initial_file_sha256':row['initial_file_sha256']})
runs={o['run'] for o in out}; vector_steps=sum(max(o['policy_steps'] for o in out if o['run']==run) for run in runs)
steps=sum(o['policy_steps'] for o in out); assert steps==54646 and len(out)==240 and len(runs)==24
cost={'valid_decision_steps':steps,'batched_policy_steps':vector_steps,'temporal_head_forward_calls':2*vector_steps,'spatial_frontend_forward_calls':vector_steps,'policy_environment_lane_advances':10*vector_steps,'finished_lane_zero_action_advances':10*vector_steps-steps,'zero_warmup_lane_advances':5*10*24,'vector_seconds':sum(e['seconds'] for e in events if e['kind']=='vector_complete'),'gpu_admission_parameter_calls':208,'gpu_admission_frontend_calls':16,'offline_ra_cpu_batches':174,'offline_ra_model_windows':2712,'training_updates':0,'engineering01_failed_import_model_environment_calls':0}
files=[D/'LOCAL_FINAL_AUDIT.json',D/'REMOTE_FINAL_AUDIT.json',D/'MECHANISM_ANALYSIS.json',D/'PAIRED_OUTCOME_SUMMARY.json',S/'SNAPSHOT.json',R/'OFFLINE_COMPARISON.json']
closure={'status':'PASS_DIAGNOSTIC_CLOSURE','observation':snap['time'],'closed_at':datetime.datetime.now().astimezone().isoformat(),'source_contract':snap['contract_sha256'],'initial_hash_binding':accepted,'initial_verification_scope':'New recorded initial hashes match previously independently accepted byte hashes; no new independent initial-file byte read in this local closure.','evidence':{str(p.relative_to(R)):sha(p) for p in files},'cost':cost,'protection_training':'NOT_STARTED_NO_VALIDATED_SELECTIVE_HARM_RULE','limitations':['Raw RGB privately retained, not visually adjudicated in scalar analysis.','No same-state action intervention or irreversible-failure labels.','Repeated-demo exploratory analysis; no significance or protection guarantee.']}
write(D/'CLOSURE_AND_COST_AUDIT.json',closure)
status=js(R/'RUN_STATUS.json');status.update(status='COMPLETE_DIAGNOSTIC_ACCEPTED_NO_PROTECTION_TRAINING',final_directory=D.name,final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),closure_sha256=sha(D/'CLOSURE_AND_COST_AUDIT.json'),mechanism_analysis='COMPLETE_DESCRIPTIVE_CAUSAL_LIMITS_RETAINED')
status['closed_loop']['status']='COMPLETE_LOCAL_ACCEPTED';status['active_root']['status']='NATURAL_EXIT0_NO_OWN_PROCESSES'
write(R/'RUN_STATUS.json',status)
write(R/'LOCAL_RECOMPUTATION.json',{'status':'PASS','closure_sha256':sha(D/'CLOSURE_AND_COST_AUDIT.json'),'episodes':240,'pairs':120,'steps':steps,'training_updates':0})
print(json.dumps(closure,ensure_ascii=False,indent=2))
