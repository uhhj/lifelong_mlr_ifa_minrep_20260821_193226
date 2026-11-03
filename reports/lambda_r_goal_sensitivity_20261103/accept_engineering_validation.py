"""Read existing completed GOAL evidence; no remote work or scientific calls."""
from pathlib import Path
import ast,datetime,hashlib,json,sys
R=Path(__file__).resolve().parent
D=Path(sys.argv[1]);P=Path(sys.argv[2])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
textsha=lambda p:hashlib.sha256(p.read_text(encoding='utf8').encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
e=js(D/'EVIDENCE.json');assert not e['processes']
for n,x in e['files'].items():assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],n
assert e['files']['validation.exit']['text'].strip()=='0'
v=json.loads(e['files']['VALIDATION.json']['text']);lock=json.loads(e['files']['LOCK_VALIDATION.json']['text'])
assert v['status']=='PASS_GOAL_FULL_PATH_BOTH_LAMBDAS'
assert v['optimizer_started']==v['optimizer_completed']==5704 and v['extra_autograd']==36
assert v['environments']==v['production_epochs']==v['target_signals']==0 and not v['production_writes']
assert {(x['lambda_R'],x['stage']) for x in v['stages']}=={(l,s) for l in [.05,.2] for s in range(6,10)} and len(v['stages'])==8
for x in v['stages']:
 assert x['batches']=={6:100,7:139,8:185,9:337}[x['stage']]
 assert x['all_rows_states_probes_decisions_exact'] and x['serialization_next_step_exact']
assert lock['status']=='PASS_ZERO_ENV_EXACT_SUBPROCESS_MUTEX' and lock['nonoverlap'] and lock['argv_cwd_sys_path_random_output_exact']
assert lock['environments']==lock['optimizer_calls']==0
assert lock['source_sha256']==textsha(R/'environment_locked.py')
assert v['lock_validation_sha256']==e['files']['LOCK_VALIDATION.json']['sha256']
dispatch=json.loads(e['files']['DISPATCH.json']['text'])
assert dispatch['validation_source_sha256']=='60a54ef7de805bd03a45f95c1648fd064a81e2550c4312bb7a37dcdac267c4fa'
assert dispatch['source_contract_sha256']==v['diagnostic_source_contract_sha256']
for n,h in v['reviewed_production_source_sha256'].items():assert textsha(R/n)==h,n
pre=js(P/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY' and not pre['processes'] and not pre['gpu']
assert all(x['headroom']>=0 for x in pre['capacity'].values())
assert pre['optimizer_calls']==pre['environments']==0
closure=js(R/'OBJECT_COMPLETION_BINDING.json');assert closure['status']=='OBJECT_ACCEPTED_PUBLISHED_HOOK_CLEANED'
assert sha(Path(closure['acceptance_path']))==closure['acceptance_sha256']
assert sha(Path(closure['cleanup_evidence']))==closure['cleanup_evidence_sha256']
base=js(R/'BASELINE_BINDING.json');bd=Path(base['final_path'])
assert sha(bd/'EVIDENCE.json')==base['evidence_sha256'] and sha(bd/'LOCAL_FINAL_AUDIT.json')==base['audit_sha256']
review=js(R/'SOURCE_REVIEW.json')
for n,x in review['files'].items():assert textsha(R/n)==x['new_sha256'] and textsha(Path(x['source']))==x['original_sha256'],n
for f in R.rglob('*.py'):ast.parse(f.read_text(encoding='utf8'),feature_version=(3,8))
# The standalone final scripts must be exactly the scripts embedded by the orchestrator.
tree=ast.parse((R/'final_readonly.py').read_text(encoding='utf8'))
embedded={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']}
assert embedded['CHECK_SCRIPT']==(R/'check_readonly.py').read_text(encoding='utf8')
for case,t in embedded['SCRIPTS'].items():assert t==(R/('final_'+case+'_readonly.py')).read_text(encoding='utf8')
for name in ['ENGINEERING_VALIDATION.json','LOCK_VALIDATION.json']:
 actual='VALIDATION.json' if name=='ENGINEERING_VALIDATION.json' else name
 assert not (R/name).exists()
 (R/name).write_bytes(e['files'][actual]['text'].encode())
out=dict(status='PASS_COMPLETE_GOAL_ENGINEERING_VALIDATION_NOT_SCIENTIFIC_ACCEPTANCE',observation=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),snapshot_sha256=sha(D/'EVIDENCE.json'),validation_sha256=sha(R/'ENGINEERING_VALIDATION.json'),lock_validation_sha256=sha(R/'LOCK_VALIDATION.json'),optimizer_started=5704,optimizer_completed=5704,extra_autograd=36,environments=0,production_epochs=0,stage_pairs=8,processes=[],validation_exit=0,production_sources_exact=len(v['reviewed_production_source_sha256']),preflight_sha256=sha(P/'EVIDENCE.json'),capacity=pre['capacity'],future_stage_admission='REQUIRED',final_wrappers='STATIC_ONLY_NOT_EXECUTED')
write(D/'LOCAL_AUDIT.json',out);write(R/'ENGINEERING_ACCEPTANCE.json',out)
write(R/'PREFLIGHT_LOCAL_AUDIT.json',dict(status=pre['status'],sha256=sha(P/'EVIDENCE.json'),observed=pre['observed'],capacity=pre['capacity'],optimizer_calls=0,environments=0))
print(json.dumps(out,ensure_ascii=False))
