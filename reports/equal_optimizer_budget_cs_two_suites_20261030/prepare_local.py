"""Compose only new C runs. Never executes previous build/deploy/training code."""
from pathlib import Path
import ast,json,hashlib,difflib,sys
R=Path(__file__).resolve().parent
OLD=R.parent/'20261027_c_goal42_43_object41_42_43'
SROOT={'GOAL':R.parent/'20261029_support_gate_full_s_goal','OBJECT':R.parent/'20261028_support_gate_full_s_object'}
NAME='C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1'
CASES=[s+'_'+str(k) for s in ['GOAL','OBJECT'] for k in [41,42,43]]
read=lambda p:p.read_text(encoding='utf8')
js=lambda p:json.loads(read(p))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
def dump(p,x):put(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def replace(s,a,b):
 assert a in s,a
 return s.replace(a,b)

pre_dir=Path(sys.argv[1]);pre=js(pre_dir/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY'
dump(R/'PREFLIGHT_LOCAL_AUDIT.json',dict(status=pre['status'],sha256=sha(pre_dir/'EVIDENCE.json'),observed=pre['observed'],capacity=pre['capacity'],verified_unique_files=pre['verified_unique_files'],new_optimizer_calls=0,environments=0))
bundle={};review={}
for case in CASES:
 suite,seed=case.split('_');seed=int(seed);sr=SROOT[suite]/('seed%d'%seed);orig=OLD/(case if case!='GOAL_41' else 'GOAL_42')
 parent=js(sr/'PARENT_BINDING.json');assert hashlib.sha256(read(sr/'PARENT_BINDING.json').encode()).hexdigest()==pre['bindings'][case]['parent_binding_sha256'],case
 parent.update(seed=seed,suite=suite)
 rows=js(pre_dir/'diagnostics'/(case+'_manifest_metadata.json'))
 for stage in range(6,10):
  rr=[x for x in rows if x['stage']==stage];assert [x['index'] for x in rr]==list(range(parent['stages'][str(stage)]['population']))
  parent['stages'][str(stage)]['rows']=[{k:x[k] for k in ['task','bc_valid','sha256']} for x in rr]
 mapping={n:read(orig/n) for n in ['train.py','runtime.py','updates.py','losses.py','legacy_losses.py','synthetic_checks.py','directional_prototype.py','evaluate.py']}
 oldparent=js(orig/'PARENT_BINDING.json')
 runtime=mapping['runtime.py'].replace(repr(oldparent['output']),repr(parent['output'])).replace(repr(oldparent['source']),repr(parent['source'])).replace("CORESET_SEED='42'", "CORESET_SEED='%d'"%seed) if case=='GOAL_41' else mapping['runtime.py']
 runtime=replace(runtime,' a.start_task(-1,loader);a.refs={}', ''' a.cfg.train.n_epochs=STAGE_EPOCHS[ds.stage]
 a.total_steps=STAGE_EPOCHS[ds.stage]*len(loader)
 a.start_task(-1,loader);a.refs={}
 fn=a.scheduler.lr_lambdas[0];total=a.total_steps
 assert a.scheduler.last_epoch==0 and fn(0)==0 and fn(total)==0
 assert abs(fn(500)-1.)<1e-12 and fn(total-1)>0 and fn(total//2)>0
 assert a.cfg.train.scheduler.kwargs.warmup_steps==500
 write(R/('SCHEDULE_C_%d.json'%ds.stage),dict(status='PASS_ACTUAL_SCHEDULE',epochs=STAGE_EPOCHS[ds.stage],batches=len(loader),total_steps=total,warmup_steps=500,base_lrs=a.scheduler.base_lrs,values={str(k):fn(k) for k in [0,1,500,total//2,total-1,total]},production_calls_before=CALLS['production'],isolated_calls_before=CALLS['admission']))''')
 runtime+='''
# Physical step counter survives candidate/state restore; it does not alter updates.
import functools
CALLS={'production':0,'admission':0}
_count_mode='admission'
_original_adamw_step=torch.optim.AdamW.step
@functools.wraps(_original_adamw_step)
def _counted_step(self,*args,**kwargs):
 result=_original_adamw_step(self,*args,**kwargs)
 CALLS[_count_mode]+=1
 return result
torch.optim.AdamW.step=_counted_step
def set_count_mode(mode):
 global _count_mode
 assert mode in CALLS;_count_mode=mode

from candidate import cpu_snapshot
_reference_cpu=cpu
_reference_capture=capture
cpu=cpu_snapshot
_copy_checked={}
def capture(a):
 result=_reference_capture(a)
 if a.scheduler.last_epoch==0 and str(a.step) not in _copy_checked:
  global cpu
  try:
   cpu=_reference_cpu;oracle=_reference_capture(a)
  finally:cpu=cpu_snapshot
  assert equal(result,oracle),'Actual C stage full-state copy mismatch'
  _copy_checked[str(a.step)]=dict(all_state_exact=True,optimizer_calls_added=0,environments_added=0)
  write(R/'COPY_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=_copy_checked,candidate_sha256=sha(R/'candidate.py')))
 return result
'''
 mapping['runtime.py']=runtime
 total=142200 if suite=='GOAL' else 195300
 layout=f'''from pathlib import Path
ARMS='C'
SEED={seed}
SUITE={suite!r}
NAME={NAME!r}
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
CASE_PRIVATE=DATA_PRIVATE/{case!r}
OUTPUT={{a:CASE_PRIVATE/a for a in ARMS}}
VOLUME_BOUND={{'/root':0,'/data':18*1024**3}}
STAGE_EPOCHS={{6:100,7:200,8:200,9:200}}
TOTAL_UPDATES={total}
POP={ {int(k):v['population'] for k,v in parent['stages'].items()}!r}
VALID={ {int(k):v['valid'] for k,v in parent['stages'].items()}!r}
OLD={ {int(k):v['old_valid'] for k,v in parent['stages'].items()}!r}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return CASE_PRIVATE/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return CASE_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
'''
 mapping['layout.py']=layout
 mapping['candidate.py']=read(SROOT['OBJECT']/'engineering_copy_20261029_v1/candidate.py')
 train=mapping['train.py']
 needle=" restore(a,initial)\n p=a.policy.module;p.eval()"
 extra=''' # Ten-step replay plus checkpointed two-step continuation: 24 isolated calls.
 if stage>6:
  restore(a,initial)
  for _ in range(10):update(a,batch,arm,cache,scale,weights,info,stage)
  ten=capture(a)
  restore(a,initial)
  for _ in range(10):update(a,batch,arm,cache,scale,weights,info,stage,legacy=True)
  assert equal(ten,capture(a))
  stream10=io.BytesIO();torch.save(ten,stream10);stream10.seek(0);loaded10=torch.load(stream10,map_location='cpu')
  assert equal(ten,loaded10)
  for _ in range(2):update(a,batch,arm,cache,scale,weights,info,stage)
  twelve=capture(a)
  restore(a,loaded10)
  for _ in range(2):update(a,batch,arm,cache,scale,weights,info,stage)
  assert equal(twelve,capture(a));calls+=24
 restore(a,initial)
 p=a.policy.module;p.eval()'''
 train=replace(train,needle,extra)
 train=replace(train,"optimizer_calls=calls,roundtrip_exact=True", "optimizer_calls=calls,ten_step_and_checkpoint_continuation_exact=stage>6,physical_calls_observed=dict(CALLS),roundtrip_exact=True")
 train=replace(train," a,b=build(c)"," a,b=build(c)") if " a,b=build(c)" in train else train
 needle=" write(R/('INITIAL_'+arm+'.json'),dict("
 train=replace(train,needle," ref=js(R/'S_REFERENCE.json');prior=js(ref['INITIAL_S.json']['path']);assert sha(ref['INITIAL_S.json']['path'])==ref['INITIAL_S.json']['sha256']\n assert tensor==prior['tensors'] and digest==prior['rng'] and hardware==prior['hardware'],'Actual accepted S initialization/RNG/hardware mismatch'\n"+needle)
 train=replace(train,"   cache,scale,weights,teacher=prepare(arm,stage,ds,a);admit(arm,stage,ds,a,cache,scale,weights,info)","   cache,scale,weights,teacher=prepare(arm,stage,ds,a)\n   set_count_mode('admission');before_admission=CALLS['admission'];admit(arm,stage,ds,a,cache,scale,weights,info)\n   assert CALLS['admission']-before_admission==(4 if stage==6 else 29)\n   set_count_mode('production')")
 train=replace(train,'   for ep in range(100):','   for ep in range(STAGE_EPOCHS[stage]):')
 train=replace(train,'ep in [0,9,49,99]','ep in [0,19,99,199]')
 train=replace(train,'row=dict(stage=stage,epoch=ep,steps=a.step,','row=dict(stage=stage,epoch=ep,steps=a.step,physical_optimizer_calls=CALLS[\'production\'],isolated_optimizer_calls=CALLS[\'admission\'],')
 train=replace(train,"    history.append(row);state=capture(a)","    assert CALLS['production']==a.step\n    history.append(row);state=capture(a)")
 train=replace(train,'history=history))','history=history,physical_calls=dict(CALLS)))')
 train=replace(train,'epoch=99,steps=a.step','epoch=STAGE_EPOCHS[stage]-1,steps=a.step')
 train=replace(train,"    same_device=equal(actual,reference);assert same_device,'Task6 must match current-card C exactly'", "    same_device=equal(actual,reference);assert same_device,'Task6 C self mismatch'\n    sr=js(R/'S_REFERENCE.json')['stage6_checkpoint'];assert sha(sr['path'])==sr['sha256']\n    accepted_s=torch.load(sr['path'],map_location='cpu')['state_dict'];assert equal(actual,accepted_s),'Task6 accepted S tensor mismatch'\n    write(R/'TASK6_ACCEPTED_S_PARITY.json',dict(status='PASS',all_tensors_exact=True,S_cp_sha256=sr['sha256'],C_cp_sha256=sha(target),new_S_calls=0))")
 train=replace(train,'epochs=100,stage=stage','epochs=STAGE_EPOCHS[stage],stage=stage')
 train=replace(train,('assert a.step==76100' if suite=='GOAL' else 'assert a.step==105500')+' and len(history)==400',"assert a.step==TOTAL_UPDATES and len(history)==700 and CALLS=={'production':TOTAL_UPDATES,'admission':91}")
 train=replace(train,"dict(status='PASS',epochs=400,steps=a.step,","dict(status='PASS',epochs=700,steps=a.step,physical_optimizer_calls=CALLS['production'],isolated_optimizer_calls=CALLS['admission'],")
 mapping['train.py']=train
 for n,s in mapping.items():
  ast.parse(s,feature_version=(3,8))
  ref=orig/n;original=read(ref) if ref.exists() else ''
  review[case+'/'+n]=dict(reference=str(ref),source_sha256=hashlib.sha256(original.encode()).hexdigest(),sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=original==s)
  if original!=s:put(R/'review'/(case+'_'+n+'.diff'),''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(ref),tofile=case+'/'+n)))
 for n in ['updates.py','losses.py','legacy_losses.py','synthetic_checks.py','directional_prototype.py','evaluate.py']:assert mapping[n]==read(orig/n)
 mapping['PARENT_BINDING.json']=json.dumps(parent,indent=2)+'\n'
 mapping['S_REFERENCE.json']=json.dumps(pre['bindings'][case]['accepted_S_reference'],indent=2)+'\n'
 mapping['PROTOCOL.md']=read(R/'PROTOCOL.md')
 for n,s in mapping.items():put(R/case/n,s);bundle[case+'/'+n]=s
for n in ['controller.py','PROTOCOL.md','S_ACCEPTANCE.json']:bundle[n]=read(R/n)
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(pre_dir/'EVIDENCE.json'),previous=pre['previous'],hardware=pre['hardware'],capacity=pre['capacity']),indent=2)+'\n'
dump(R/'DEPLOY_TEXT_BUNDLE.json',bundle)
dump(R/'UPLOAD_REVIEW.json',dict(status='REVIEWED_TEXT_ONLY_PENDING_ACTUAL_IMPORT_AND_ADMISSION',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},source_review=review,new_S_runs=0))
print(json.dumps(dict(status='BUILT_C_ONLY_NOT_DISPATCHED',cases=CASES,files=len(bundle),epochs=4200,production_calls=1012500,isolated_calls=546,formal_episodes=1200)))
