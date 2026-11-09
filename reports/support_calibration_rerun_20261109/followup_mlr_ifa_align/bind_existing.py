"""Bind the six accepted Random baselines and ALIGN source; no model/data loading."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;W=R.parents[2]
sources={}
def read(p):
 p=W/p;raw=p.read_bytes();sources[p.relative_to(W).as_posix()]=hashlib.sha256(raw).hexdigest();return raw.decode('utf-8-sig')
def js(p):return json.loads(read(p))
audit=js(Path('execution/20261107_mlr_ifa_reproduction_audit/AUDIT.json'))
table=js(Path('execution/20261109_random_kcenter_ra_tables/RESULTS.json'))
assert table['status']=='PASS_18_RUN_READONLY_RECOMPUTATION'
cases={}
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  key=f'{suite}_{seed}';old=audit['results'][key];root=Path('execution')/old['root']
  accept=js(root/'INDEPENDENT_ACCEPTANCE.json');assert accept['status']=='PASS'
  modal=js(root/'MODAL_CONTRACT.json');assert modal['flags']==dict(vision=True,language=True,state=True)
  for name,path in old['source_binding'].items():
   read(Path(path));assert sources[Path(path).as_posix()]==modal['source_sha256']['coreset_exp/'+name]
  initial=js(root/'random/INITIAL_STATE.json');complete=js(root/'random/TRAINING_COMPLETE.json')
  epochs=js(root/'random/epochs.json');assert len(epochs)==400 and complete['epochs']==400 and complete['steps']==(105500 if suite=='OBJECT' else 76100)
  cfg_path=root/'resolved_random.json'
  if not (W/cfg_path).exists():cfg_path=Path('execution/20261016_seed43_three_arm_repeat_v1/launch_evidence/rc/resolved_random.json')
  cfg=js(cfg_path);assert cfg['seed']==seed and cfg['gamma']==(.3 if suite=='OBJECT' else .7)
  run=next(x for x in table['runs'] if x['suite']==suite and x['seed']==seed and x['method']=='Random')
  records=0
  for p in run['sources']:
   rows=[json.loads(x) for x in read(Path(p)).splitlines() if x.strip()]
   assert len(rows)==20 and sorted(x['initial_state_index'] for x in rows)==list(range(20))
   assert all(x['seed']==seed for x in rows)
   assert sources[Path(p).as_posix()]==table['source_sha256'][Path(p).as_posix()]
   records+=len(rows)
  assert records==200
  cases[key]=dict(parent_light_root=root.as_posix(),baseline_records=records,baseline_metrics=run['metrics'],baseline_matrix=run['matrix'],training_source=old['source_binding'],config=cfg_path.as_posix(),official_checkpoint_sha256=old['pretrain_sha256'],historical_gpu=old['gpu'],initial_state_sha256=sources[(root/'random/INITIAL_STATE.json').as_posix()],baseline_retrained=False,baseline_reevaluated=False,random_members_remote_verification='REQUIRED_BEFORE_DEPLOYMENT')
for p in ['execution/20261022_align_object_extension_goal_conditional/PROTOCOL.md','execution/20261022_align_object_extension_goal_conditional/bc_labels.py','execution/20261022_align_object_extension_goal_conditional/additions/caet_v3_exp/model.py','execution/20261022_align_object_extension_goal_conditional/additions/caet_v3_exp/observe.py','execution/20261107_align_cqs_audit/REPORT.md']:
 read(Path(p))
for p,h in sources.items():assert hashlib.sha256((W/p).read_bytes()).hexdigest()==h
out=dict(status='PASS_READONLY_BASELINE_AND_ALIGN_SOURCE_BINDING_NOT_NEW_RUN_ACCEPTANCE',observed=datetime.datetime.now().astimezone().isoformat(),cases=cases,existing_baseline_records=1200,new_training=0,new_forward=0,new_environment=0,source_sha256=sources)
(R/'BASELINE_BINDING.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
authorization=dict(status='AUTHORIZED_FOLLOWUP_PENDING_IMPLEMENTATION_AND_FRESH_CAPACITY',observed=out['observed'],user_confirmed=True,suites=['OBJECT','GOAL'],seeds=[41,42,43],runs=6,method='MLR_IFA_RANDOM_PLUS_ALIGN_ONLY',reuse_existing_baseline=True,start_after='Current q=1 OBJECT and GOAL six-run completion, independent acceptance and publication',current_coordinator_modified=False,new_run_deployed=False,new_run_started=False,protocol=str(R/'PROTOCOL.md'),protocol_sha256=hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest(),baseline_binding_sha256=hashlib.sha256((R/'BASELINE_BINDING.json').read_bytes()).hexdigest(),pending=['Adapt original Random training with accepted ALIGN target/mask only','Verify frozen Random members/order and next-action label identity','Review own GPU admission and original evaluation path','Fresh capacity after q1 output growth; no old scientific deletion or local payload archive','Unique deployment after current authorized queue has completed'])
(R.parent/'AUTHORIZED_MLR_IFA_ALIGN_FOLLOWUP.json').write_text(json.dumps(authorization,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=authorization['status'],bound_cases=len(cases),baseline_records=1200,source_files=len(sources),new_training=0,new_environment=0)))
