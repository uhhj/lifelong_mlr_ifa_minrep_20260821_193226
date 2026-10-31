"""Compose new Q only from accepted full S; no previous entry point is run."""
from pathlib import Path
import json,hashlib,ast,difflib,sys
R=Path(__file__).resolve().parent;P=R.parent
NAME='FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1'
ROOTS={'GOAL':P/'20261029_support_gate_full_s_goal','OBJECT':P/'20261028_support_gate_full_s_object'}
CASES=[s+'_'+str(k) for s in ROOTS for k in (41,42,43)]
read=lambda p:p.read_text(encoding='utf8')
js=lambda p:json.loads(read(p))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
def dump(p,x):put(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rep(s,a,b):assert a in s,a;return s.replace(a,b)
prepath=Path(sys.argv[1]);pre=js(prepath/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY'
assert not (R/'DEPLOY_TEXT_BUNDLE.json').exists(),'No rebuild of an existing bundle'
binding=js(R/'S_ACCEPTANCE.json')
for suite,x in binding['sources'].items():
 d=Path(x['final_path']);assert sha(d/'EVIDENCE.json')==x['evidence_sha256'] and sha(d/'LOCAL_FINAL_AUDIT.json')==x['audit_sha256']
dump(R/'PREFLIGHT_LOCAL_AUDIT.json',dict(status=pre['status'],sha256=sha(prepath/'EVIDENCE.json'),observed=pre['observed'],capacity=pre['capacity'],verified_unique_files=pre['verified_unique_files'],optimizer_calls=0,environments=0))
bundle={};review={}
for case in CASES:
 suite,seed=case.split('_');seed=int(seed);sr=ROOTS[suite]/('seed%d'%seed)
 parent=js(sr/'PARENT_BINDING.json');assert sha(sr/'PARENT_BINDING.json')==pre['bindings'][case]['parent_binding_sha256']
 ref=pre['bindings'][case]['accepted_S_reference'];assert ref==binding['cases'][case]['source_S_reference']
 parent.update(suite=suite,seed=seed)
 total=sum(x['batches']*100 for x in parent['stages'].values());assert total==(76100 if suite=='GOAL' else 105500)
 names=['runtime.py','updates.py','legacy_updates.py','memory.py','legacy_memory.py','support_memory.py','losses.py','legacy_losses.py','synthetic_checks.py','fork_tools.py','gate_admission.py','initialization.py','collect.py','evaluate.py','train.py','layout.py']
 mapping={n:read(sr/n) for n in names}
 layout=mapping['layout.py'];old=next(ast.literal_eval(n.value) for n in ast.parse(layout).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME' for t in n.targets))
 layout=rep(layout,"ARMS='S'","ARMS='Q'").replace(old,NAME)
 layout=rep(layout,"DATA_PRIVATE/('seed%d'%SEED)/a",'DATA_PRIVATE/'+repr(case)+'/a')
 layout=layout.replace("'/data':8*1024**3","'/data':8*1024**3")
 mapping['layout.py']=layout+'\nTOTAL_UPDATES='+str(total)+'\nSUITE='+repr(suite)+'\n'
 runtime=mapping['runtime.py'].split('# Independent snapshots:')[0]
 runtime+='''
# Same tested copy optimization, verified at every actual stage start.
from candidate import cpu_snapshot
_reference_cpu=cpu
_reference_capture=capture
_copy_checked={}
cpu=cpu_snapshot
def capture(a):
 result=_reference_capture(a)
 if a.scheduler.last_epoch==0 and str(a.step) not in _copy_checked:
  global cpu
  try:
   cpu=_reference_cpu;oracle=_reference_capture(a)
  finally:cpu=cpu_snapshot
  assert equal(result,oracle)
  _copy_checked[str(a.step)]=dict(all_state_exact=True,optimizer_calls_added=0,environments_added=0)
  write(R/'COPY_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=_copy_checked,candidate_sha256=sha(R/'candidate.py')))
 return result
import functools
CALLS={'production':0,'admission':0}
_count_mode='admission'
_original_adamw_step=torch.optim.AdamW.step
@functools.wraps(_original_adamw_step)
def _counted_step(self,*args,**kwargs):
 result=_original_adamw_step(self,*args,**kwargs);CALLS[_count_mode]+=1;return result
torch.optim.AdamW.step=_counted_step
def set_count_mode(mode):
 global _count_mode
 assert mode in CALLS;_count_mode=mode
'''
 runtime=rep(runtime,' a.start_task(-1,loader);a.refs={}', ''' a.start_task(-1,loader);a.refs={}
 assert a.cfg.train.n_epochs==100 and a.total_steps==100*len(loader)
 fn=a.scheduler.lr_lambdas[0];assert fn(0)==0 and fn(500)==1 and fn(a.total_steps)==0 and fn(a.total_steps-1)>0
 write(R/('SCHEDULE_Q_%d.json'%ds.stage),dict(status='PASS_ACTUAL_SCHEDULE',epochs=100,batches=len(loader),total_steps=a.total_steps,warmup_steps=500))''')
 mapping['runtime.py']=runtime
 mapping['candidate.py']=read(ROOTS['OBJECT']/'engineering_copy_20261029_v1/candidate.py')
 init=mapping['initialization.py'];needle=" write(R/('INITIAL_'+arm+'.json'),dict("
 init=rep(init,needle," ref=js(R/'S_REFERENCE.json')['INITIAL_S.json'];assert sha(ref['path'])==ref['sha256'];old=js(ref['path'])\n assert tensor==old['tensors'] and digest==old['rng'] and hardware==old['hardware'],'Accepted S initialization/RNG/hardware mismatch'\n"+needle)
 mapping['initialization.py']=init
 mapping['evaluate.py']=rep(mapping['evaluate.py'],"p=load('S',6,c,shape)","p=load('Q',6,c,shape)")
 t=mapping['train.py']
 t=rep(t,'            if stage==6:\n                admit(',"            set_count_mode('admission');admission_before=CALLS['admission']\n            if stage==6:\n                admit(")
 t=rep(t,"pr,'G')","pr,'Q')").replace("method='support_gate_full_S'","method='support_weighted_Q_no_gate'")
 t=rep(t,'            for ep in range(100):',"            assert CALLS['admission']-admission_before==(4 if stage==6 else 29)\n            set_count_mode('production')\n            for ep in range(100):")
 start=t.index('                        base=capture(a)');end=t.index('                    else:\n                        rows=',start)
 t=t[:start]+'''                        rows=[update(a,b,'Q',memory,scale,info,stage) for b in block]
                        retained_steps+=len(block);physical+=len(block)
'''+t[end:]
 t=rep(t,'                assert bi==len(loader)',"                assert bi==len(loader) and CALLS['production']==physical==a.step\n                assert len(gates)==0 and not (R/'GATES.jsonl').exists()")
 t=t.replace('physical_optimizer_calls=physical,gate_blocks=',"physical_optimizer_calls=physical,physical_calls_observed=dict(CALLS),gate_blocks=")
 needle="                write(R/('TASK6_PARITY_'+arm+'.json'),dict("
 t=rep(t,needle,"                sr=js(R/'S_REFERENCE.json')['stage6_checkpoint'];assert sha(sr['path'])==sr['sha256']\n                assert equal(actual,torch.load(sr['path'],map_location='cpu')['state_dict']),'Task6 accepted S tensor mismatch'\n"+needle)
 t=t.replace("same_seed_new_arm_comparator='NOT_RUN_SINGLE_S'","same_seed_new_arm_comparator='ACCEPTED_S_ALL_TENSORS_EXACT',accepted_S_cp_sha256=sr['sha256']")
 oldassert=next(line for line in t.splitlines() if 'assert a.step==' in line and 'len(history)' in line)
 t=rep(t,oldassert,"        assert a.step==TOTAL_UPDATES and len(history)==400 and physical==TOTAL_UPDATES and not gates\n        assert CALLS=={'production':TOTAL_UPDATES,'admission':91}")
 mapping['train.py']=t
 for n,s in mapping.items():
  ast.parse(s,feature_version=(3,8));original=read(sr/n) if (sr/n).exists() else read(ROOTS['OBJECT']/'engineering_copy_20261029_v1/candidate.py')
  review[case+'/'+n]=dict(source=str(sr/n),original_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==original)
  if original!=s:put(R/'review'/(case+'_'+n+'.diff'),''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(sr/n),tofile=case+'/'+n)))
 for n in ['updates.py','legacy_updates.py','memory.py','legacy_memory.py','support_memory.py','losses.py','legacy_losses.py','synthetic_checks.py','fork_tools.py','gate_admission.py','collect.py']:assert mapping[n]==read(sr/n)
 mapping.update({'PARENT_BINDING.json':json.dumps(parent,indent=2)+'\n','INITIAL_SPLIT.json':read(sr/'INITIAL_SPLIT.json'),'S_REFERENCE.json':json.dumps(ref,indent=2)+'\n','PROTOCOL.md':read(R/'PROTOCOL.md')})
 for n,s in mapping.items():put(R/case/n,s);bundle[case+'/'+n]=s
for n in ['controller.py','PROTOCOL.md']:bundle[n]=read(R/n)
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(prepath/'EVIDENCE.json'),previous=pre['previous'],capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
dump(R/'DEPLOY_TEXT_BUNDLE.json',bundle)
dump(R/'UPLOAD_REVIEW.json',dict(status='STATIC_REVIEW_PENDING_OWN_GPU_ADMISSION',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},source_review=review,private_payloads_included=False))
print(json.dumps(dict(status='BUILT_NOT_DISPATCHED',files=len(bundle),epochs=2400,production_calls=544800)))
