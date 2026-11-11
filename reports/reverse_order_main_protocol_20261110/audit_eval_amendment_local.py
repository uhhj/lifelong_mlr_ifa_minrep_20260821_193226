"""Verify the actual protocol amendment; no model, training, environment or SSH."""
from pathlib import Path
import ast,hashlib,json,sys,datetime
R=Path(__file__).resolve().parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=Path(sys.argv[1]);e=J(d/'EVIDENCE.json');m=J(R/'EVALUATION_PROTOCOL_AMENDMENT.json');c=J(R/'CONTRACT.json');s=J(R/'evaluation_unification_v1/STATIC_REVIEW.json')
assert e['status']=='APPLIED_USER_AUTHORIZED_COMMON_DEBR_EVALUATION' and e['controller_resumed']
assert e['new_contract_sha256']==H(R/'CONTRACT.json')==s['new_contract_sha256']
assert e['parent_contract_sha256']==H(R/'evaluation_unification_v1/original/CONTRACT.json')==m['parent_root_contract_sha256']
assert e['training_case_contracts_before']==e['training_case_contracts_after']
for name,v in e['training_case_contracts_after'].items():assert H(R/'cases'/name/'CONTRACT.json')==v
assert s['MLR_training_branch_AST_equal'] and s['entire_DEBR_branch_AST_equal']
for p,v in c['files'].items():assert H(R/p)==v,p
for p,v in m['evaluation_source_sha256'].items():assert H(R/p)==v,p
parent=(R/m['template_parent']).read_text(encoding='utf-8-sig')
actual=(R/'evaluation_unification_v1/evaluate.py').read_text(encoding='utf-8-sig')
assert parent.replace('from runtime import *','from unified_mlr_runtime import *').replace("p=load('S',6,c,shape)","p=load('MLR',6,c,shape)")==actual
assert len(e['CPU_admission'])==6
loaded=forwards=0
for name,x in e['CPU_admission'].items():
    assert x['status']=='PASS_CPU_IMPORT_CONFIG_AND_CHECKPOINT_COMPATIBILITY'
    assert x['scientific_eval_config_equal'] and not x['CUDA_initialized']
    assert x['new_optimizer']==x['new_environment']==0
    assert all(v['AST_equal'] for v in x['architecture'].values())
    for st,v in x['completed_checkpoints'].items():
        assert v['strict_keys']==100 and v['parameters']==21633611
        loaded+=1
    forwards+=x['CPU_policy_forward_calls']
assert loaded==8 and forwards==16
assert not e['formal_started_before_change'] and not e['completed_training_restarted'] and not e['new_controller_launched']
result=dict(status='PASS_ACTUAL_EVALUATION_AMENDMENT_PENDING_FORMAL_GPU_ADMISSION_AND_FINAL_ACCEPTANCE',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),contract=e['new_contract_sha256'],parent_contract=e['parent_contract_sha256'],case_training_contracts_unchanged=12,completed_MLR_checkpoints_strict_CPU=loaded,CPU_policy_forwards=forwards,CPU_config_import_cases=6,copied_DEBR_evaluator_changes_only=['runtime import','arm name'],transaction_pause_seconds=e['transaction_pause_seconds'],new_training=0,new_environment=0,first_failed_probe='install_eval_amendment_20261111_093327/ATTEMPT.json; remote evaluation_unification_v1_failed_stage01 retained',source_evidence_sha256=H(d/'EVIDENCE.json'),formal_cache_parity='Must pass each actual MLR GPU evaluator admission before any environment episode')
(R/'evaluation_unification_v1/LOCAL_AMENDMENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
