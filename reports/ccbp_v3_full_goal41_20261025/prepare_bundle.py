"""Local-only fresh preflight acceptance and exact reviewed deployment bundle."""
import ast, hashlib, json, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'20261024_ccbp_v2_full'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
final=P/'final_readonly_20261025_054702'
closure=json.loads((final/'CLOSURE_INDEX.json').read_text(encoding='utf8'))
assert closure['status']=='PASS_COMPLETE_BOUND_CLOSURE'
for n,m in closure['evidence_files'].items():
    p=P/n;assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes'],n
evidence=json.loads((final/'EVIDENCE.json').read_text(encoding='utf8'))
audit=json.loads((final/'LOCAL_FINAL_AUDIT.json').read_text(encoding='utf8'))
assert audit['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and sha(final/'EVIDENCE.json')==audit['evidence_sha256']
parent=dict(status=audit['status'],root='/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02',contract=closure['contract'],closure_sha256=sha(final/'CLOSURE_INDEX.json'),local_audit_sha256=sha(final/'LOCAL_FINAL_AUDIT.json'),evidence_sha256=sha(final/'EVIDENCE.json'),strict=evidence['strict'],published_commit='decaa233260d7056e5c343b6dd35ebf5394d2dfa',scientific_main_exploratory_criterion=False)
(R/'PARENT_LOCAL_ACCEPTANCE.json').write_text(json.dumps(parent,indent=2)+'\n',encoding='utf8')
if len(sys.argv)<2:
    print('Parent closure bound locally; no new remote preflight or deployment bundle created.')
    raise SystemExit(0)
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
assert v['status']=='PASS_PREDEPLOY_READONLY' and not v['gpu'] and not v['processes']
assert v['parent_contract']==parent['contract'] and time.time()-v['observed']<1800
for arm in 'BCNV':
    for stage in range(6,10):assert v['parent_stage_checkpoints'][arm+str(stage)]['sha256']==parent['strict'][arm][str(stage)]['sha256']
assert all(x['headroom']>=0 for x in v['volumes'].values())
assert json.loads((R/'INDEPENDENT_CPU_REVIEW.json').read_text())['status']=='PASS_SYNTHETIC_NOT_REAL_ADMISSION'
assert json.loads((R/'cpu_review/synthetic_test_results.json').read_text())['status']=='PASS_LOCAL_SYNTHETIC_ONLY'
accept=dict(status='PASS_LOCAL_PREDEPLOY_ACCEPTANCE',remote_observed=v['observed'],parent_contract=parent['contract'],preflight_evidence_sha256=sha(D/'EVIDENCE.json'),parent_local_acceptance_sha256=sha(R/'PARENT_LOCAL_ACCEPTANCE.json'),capacity=v['volumes'],hardware=v['hardware'],actual_GPU_admission='PENDING',controller='NOT_DISPATCHED')
(R/'PREDEPLOY_ACCEPTANCE.json').write_text(json.dumps(accept,indent=2)+'\n',encoding='utf8')
names=['runtime.py','layout.py','losses.py','legacy_losses.py','directional_prototype.py','synthetic_checks.py','updates.py','train.py','evaluate.py','controller.py','PROTOCOL.md','PARENT_LOCAL_ACCEPTANCE.json','PREDEPLOY_ACCEPTANCE.json']
bundle={n:(R/n).read_text(encoding='utf8') for n in names}
for n,s in bundle.items():
    if n.endswith('.py'):ast.parse(s)
(R/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=True,indent=2)+'\n',encoding='utf8')
print(json.dumps(accept,indent=2))
