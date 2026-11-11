"""BUNDLE is a reviewed, code/metadata-only literal prepended by local builder."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,traceback
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0,str(Q))
from reverse_common import verify,verify_case,read,write,processes,capacity
H=lambda b:hashlib.sha256(b).hexdigest()
parent=BUNDLE['parent_contract_sha256'];newsha=BUNDLE['new_contract_sha256']
assert H((Q/'CONTRACT.json').read_bytes())==parent
verify();capacity()
assert not (Q/'EVALUATION_PROTOCOL_AMENDMENT_APPLIED.json').exists(),'Amendment already applied; no repeat'
assert not (Q/'controller.exit').exists() and not (Q/'FAILURE.json').exists()
spec=read(Q/'SPEC.json')
def no_formal_started():
 for name,entry in spec['cases'].items():
  r=Q/'cases'/name
  assert not any((r/f).exists() for f in ['EVAL_DISPATCH.json','EVALUATION_STARTED.json','NORMALIZED_OUTCOMES.json','OUTCOMES.json']),name
  if entry['method']=='MLR':
   inp=read(r/'INPUT.json');native=Path(inp['output_root'])/'evaluation'
   assert not list(native.glob('trials*.jsonl')) and not list(native.glob('attempts*.jsonl')),name
no_formal_started()
staged_failure=Q/'evaluation_unification_v1/probe_attempts/OBJECT_MLR_41.json'
if staged_failure.exists():
 # The first, never-activated candidate failed its CPU shape-metadata check.
 # Preserve that complete candidate and actual failure before staging the fix.
 assert read(staged_failure)['exit']!=0
 assert read(Q/'evaluation_unification_v1/STAGING.json')['new_contract_sha256']=='c661fac5f32f6f47e2589fa0a39c5845bfd7f0d019ad417e2dd1507cca3c4543'
 retired=Q/'evaluation_unification_v1_failed_stage01'
 assert not retired.exists()
 os.rename(Q/'evaluation_unification_v1',retired)
 os.replace(Q/'EVALUATION_PROTOCOL_AMENDMENT.json',retired/'EVALUATION_PROTOCOL_AMENDMENT.json')
before_contracts={name:H((Q/'cases'/name/'CONTRACT.json').read_bytes()) for name in spec['cases']}
hot=['run_case.py','final_cpu_case.py','check_readonly.py'];old={name:(Q/name).read_bytes() for name in hot};oldcontract=(Q/'CONTRACT.json').read_bytes()
for name,text in BUNDLE['files'].items():
 target=Q/name
 assert Q.resolve() in target.resolve().parents,name
 if name in hot:continue
 raw=text.encode();target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert target.read_bytes()==raw,('Existing staging file differs; retain and version before replacing',name)
 else:
  with target.open('xb') as f:f.write(raw)
write(Q/'evaluation_unification_v1/STAGING.json',dict(status='STAGED_NOT_ACTIVE',parent_contract_sha256=parent,new_contract_sha256=newsha,time=time.time(),new_optimizer=0,new_environment=0))
probe_results={};attempts=Q/'evaluation_unification_v1/probe_attempts';attempts.mkdir(exist_ok=True)
for name,entry in spec['cases'].items():
 if entry['method']!='MLR':continue
 dst=attempts/(name+'.json');assert not dst.exists(),'Preserve previous actual CPU probe'
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl')
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B',str(Q/'evaluation_unification_v1/probe_case.py'),name],cwd=Q,env=env,capture_output=True,text=True)
 write(dst,dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr,time=time.time(),probe_sha256=H((Q/'evaluation_unification_v1/probe_case.py').read_bytes())))
 assert p.returncode==0,('CPU admission failed; original queue unchanged',name,p.stderr[-3500:])
 result=json.loads(p.stdout.strip().splitlines()[-1]);assert result['status']=='PASS_CPU_IMPORT_CONFIG_AND_CHECKPOINT_COMPATIBILITY'
 assert not result['CUDA_initialized'] and result['new_optimizer']==result['new_environment']==0
 probe_results[name]=result
write(Q/'evaluation_unification_v1/CPU_ADMISSION.json',dict(status='PASS_ALL_SIX_MLR_CPU_COMPATIBILITY',cases=probe_results,new_optimizer=0,new_environment=0))
assert H((Q/'CONTRACT.json').read_bytes())==parent
verify();no_formal_started()
pid=119466;proc=Path('/proc')/str(pid)
assert [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')]
assert proc.joinpath('cwd').resolve()==Q
assert (proc/'stat').read_text().split(') ',1)[1].split()[0] not in ['T','Z']
assert read(Q/'STATUS.json')['status']=='PARTIAL_TRAINING'
paused=False;committed=False;start=None
try:
 os.kill(pid,signal.SIGSTOP);paused=True;start=time.monotonic()
 for unused in range(100):
  if (proc/'stat').read_text().split(') ',1)[1].split()[0]=='T':break
  time.sleep(.01)
 assert (proc/'stat').read_text().split(') ',1)[1].split()[0]=='T'
 no_formal_started()
 for name in hot:
  target=Q/name;temp=target.with_name(target.name+'.user_eval_amendment_tmp')
  with temp.open('xb') as f:f.write(BUNDLE['files'][name].encode())
  os.replace(temp,target)
 temp=Q/'CONTRACT.json.user_eval_amendment_tmp'
 with temp.open('xb') as f:f.write(BUNDLE['contract_text'].encode())
 os.replace(temp,Q/'CONTRACT.json')
 assert H((Q/'CONTRACT.json').read_bytes())==newsha
 verify()
 after_contracts={name:H((Q/'cases'/name/'CONTRACT.json').read_bytes()) for name in spec['cases']}
 assert after_contracts==before_contracts
 for name in spec['cases']:verify_case(Q/'cases'/name)
 receipt=dict(status='APPLIED_USER_AUTHORIZED_COMMON_DEBR_EVALUATION',observed=time.time(),parent_contract_sha256=parent,new_contract_sha256=newsha,controller=pid,training_case_contracts_before=before_contracts,training_case_contracts_after=after_contracts,training_case_contracts_unchanged=True,MLR_training_unchanged=True,DEBR_science_unchanged=True,completed_training_restarted=False,new_controller_launched=False,formal_started_before_change=False,extra_formal_episodes=0,direct_optimizer_calls=0,direct_environment_calls=0,CPU_admission=probe_results,transaction_pause_seconds=time.monotonic()-start)
 write(Q/'EVALUATION_PROTOCOL_AMENDMENT_APPLIED.json',receipt)
 committed=True
except BaseException:
 if paused and not committed:
  for name,raw in old.items():
   temp=(Q/name).with_name(name+'.rollback_tmp');temp.write_bytes(raw);os.replace(temp,Q/name)
  temp=Q/'CONTRACT.json.rollback_tmp';temp.write_bytes(oldcontract);os.replace(temp,Q/'CONTRACT.json');verify()
 write(Q/'evaluation_unification_v1/APPLY_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),original_contract_restored=not committed))
 raise
finally:
 if paused:os.kill(pid,signal.SIGCONT)
assert committed
for unused in range(100):
 state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
 if state not in ['T','t']:break
 time.sleep(.01)
assert state not in ['T','t']
receipt.update(controller_resumed=True,controller_state=state,processes_after=processes())
write(Q/'EVALUATION_PROTOCOL_AMENDMENT_APPLIED.json',receipt)
print(json.dumps(receipt))
