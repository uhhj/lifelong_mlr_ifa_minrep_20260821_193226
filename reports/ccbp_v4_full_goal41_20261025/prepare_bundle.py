"""Bind completed reference, fresh preflight, and only reviewed text sources."""
import ast,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'20261025_ccbp_v3_full';F=P/'final_readonly_20261025_181226'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def write(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=True,indent=2)+'\n',encoding='utf8')
closure=js(F/'CLOSURE_INDEX.json');assert closure['no_remaining_experiment_work']
for n,m in closure['files'].items():
 p=P/n;assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],n
v=js(F/'EVIDENCE.json');a=js(F/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
write('PARENT_LOCAL_ACCEPTANCE.json',dict(status=a['status'],contract=closure['contract'],strict=v['strict'],closure_sha256=sha(F/'CLOSURE_INDEX.json'),audit_sha256=sha(F/'LOCAL_FINAL_AUDIT.json'),evidence_sha256=sha(F/'EVIDENCE.json'),reference_only=True))
records=json.loads(v['light']['OUTCOMES.json']['text']);records=[r for r in records if r['arm']=='C'];assert len(records)==200
write('REFERENCE_C.json',dict(records=records,parent_contract=closure['contract'],source_evidence_sha256=sha(F/'EVIDENCE.json'),not_interaction_matched=True))
D=Path(sys.argv[1]);pre=js(D/'EVIDENCE.json')
assert pre['status']=='PASS_PREDEPLOY_READONLY' and not pre['gpu'] and not pre['processes']
assert pre['parent_contract']==closure['contract'] and time.time()-pre['observed']<1800
assert all(x['headroom']>=0 for x in pre['capacity'].values())
assert pre['synthetic']['status']=='PASS_SYNTHETIC_ONLY'
for n,h in pre['reviewed_source_sha256'].items():assert hashlib.sha256((R/(n+'.py')).read_text(encoding='utf8').encode()).hexdigest()==h
write('INITIAL_SPLIT.json',pre['initial_split'])
write('PREDEPLOY_ACCEPTANCE.json',dict(status='PASS_LOCAL_PREDEPLOY_ACCEPTANCE',remote_observed=pre['observed'],parent_contract=closure['contract'],preflight_sha256=sha(D/'EVIDENCE.json'),capacity=pre['capacity'],synthetic=pre['synthetic'],actual_GPU_admission='PENDING',controller='NOT_DISPATCHED'))
names=['runtime.py','layout.py','legacy_losses.py','losses.py','memory.py','updates.py','initialization.py','train.py','collect.py','evaluate.py','controller.py','synthetic_checks.py','PROTOCOL.md','PARENT_LOCAL_ACCEPTANCE.json','PREDEPLOY_ACCEPTANCE.json','INITIAL_SPLIT.json','REFERENCE_C.json']
bundle={n:(R/n).read_text(encoding='utf8') for n in names}
for n,s in bundle.items():
 if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
write('DEPLOY_TEXT_BUNDLE.json',bundle)
write('UPLOAD_REVIEW.json',dict(status='REVIEWED_TEXT_ONLY',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},excluded='credentials, session control, private payloads, unreviewed archives'))
print('Parent closure, reference C, initial split and exact text bundle bound; no deployment.')
