"""Independent prefix accounting; never final scientific acceptance."""
from pathlib import Path
import json,hashlib,sys,math,datetime
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def js(n,default=None):return json.loads(v['light'][n]['text']) if n in v['light'] else default
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],n
 target=D/'light'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(x['text'],encoding='utf8')
assert not v['canceled_mechanism_processes']
active=R/'ACTIVE_ROOT.json'
if active.exists():assert v['contract_sha256']==json.loads(active.read_text(encoding='utf8'))['contract']
counts={};epochs=accepted=physical=admission_calls=admission_autograd=formal=collection=0;failures={}
for case in ['L005_41', 'L005_42', 'L005_43', 'L020_41', 'L020_42', 'L020_43']:
 seed=int(case.split('_')[1]);prefix=case+'/';p=json.loads((R/prefix/'PARENT_BINDING.json').read_text(encoding='utf8'))
 h=js(prefix+'EPOCHS_S.json',[]);gates=[]
 for stage in (7,8,9):
  n=prefix+'GATES_stage%d.jsonl'%stage
  if n in v['light']:gates.extend(json.loads(line) for line in v['light'][n]['text'].splitlines())
 gb={};last_key=None
 for g in gates:
  st,ep,bi=g['stage'],g['epoch'],g['batch'];nb=(p['stages'][str(st)]['population']+9)//10
  expected=sum((p['stages'][str(t)]['population']+9)//10*100 for t in range(6,st))+ep*nb+bi
  assert 0<=ep<100 and bi%10==0 and 0<=bi<nb
  assert g['steps']==min(10,nb-bi) and g['start_step']==expected and g['end_step']==expected+g['steps']
  key=(st,ep,bi);assert last_key is None or key>last_key;last_key=key
  cc,qq=g['C'],g['Q'];checks=dict(bc_new=qq['bc_new']<=cc['bc_new'],exec_new=qq['exec_new']<=cc['exec_new'],retain_improved=qq['retain_support']<cc['retain_support'])
  assert checks==g['checks'] and g['accepted']==('Q' if all(checks.values()) else 'C')
  assert all(math.isfinite(x[k]) for x in [cc,qq] for k in ['bc_new','exec_new','retain_support'])
  gb.setdefault((st,ep),[]).append(g)
 acc=phys=blocks=qaccept=0
 for i,row in enumerate(h):
  st=6+i//100;ep=i%100;spec=p['stages'][str(st)];nb=(spec['population']+9)//10
  acc+=nb;phys+=nb*(2 if st>6 else 1)
  assert row['stage']==st and row['epoch']==ep and row['batches']==nb and row['steps']==acc and row['physical_optimizer_calls']==phys
  tot=row['totals'];assert tot['valid']==spec['valid'] and tot['invalid']==spec['population']-spec['valid'] and tot['old_eligible']==spec['old_valid']
  gs=gb.get((st,ep),[]);assert len(gs)==((nb+9)//10 if st>6 else 0)
  assert [g['batch'] for g in gs]==(list(range(0,nb,10)) if st>6 else [])
  retained=sum(g['steps'] for g in gs if g['accepted']=='Q');blocks+=len(gs);qaccept+=sum(g['accepted']=='Q' for g in gs)
  assert tot['anchor_slots']==retained*4 and tot['anchor_slots']==tot['anchor_actual']+tot['anchor_empty']
  assert row['gate_blocks']==blocks and row['gate_Q_accepts']==qaccept
  assert row['seconds']>0
  assert row['physical_calls_observed']==dict(production=phys,admission=4+29*(st-6))
 if h:
  for stage in range(6,h[-1]['stage']+1):assert prefix+'ADMISSION_S_%d.json'%stage in v['light']
 if h:
  init=js(prefix+'INITIAL_S.json');sref=json.loads((R/prefix/'S_REFERENCE.json').read_text(encoding='utf8'))
  suite='OBJECT';seed=case.split('_')[1];sb=json.loads((R/'S_ACCEPTANCE.json').read_text(encoding='utf8'))['sources'][suite]
  se=json.loads((Path(sb['final_path'])/'EVIDENCE.json').read_text(encoding='utf8'));si=json.loads(se['light']['seed'+seed+'/INITIAL_S.json']['text'])
  assert all(init[k]==si[k] for k in ['tensors','rng'])
  if init['hardware']!=si['hardware']:
   migration=js(prefix+'HARDWARE_MIGRATION.json');assert migration['status']=='USER_AUTHORIZED_RTX3090' and migration['prior']==si['hardware'] and migration['current']==init['hardware'] and migration['initial_tensor_rng_exact']
  for st in range(6,h[-1]['stage']+1):
   sch=js(prefix+'SCHEDULE_S_%d.json'%st);assert sch['epochs']==100 and sch['total_steps']==100*p['stages'][str(st)]['batches'] and sch['warmup_steps']==500
  if sum(x['stage']==6 for x in h)==100 and prefix+'TASK6_PARITY_S.json' in v['light']:
   comparison=js(prefix+'TASK6_PARITY_S.json')['same_seed_new_arm_comparator']
   if case=='L005_41':assert comparison=='ACCEPTED_S_ALL_TENSORS_EXACT'
   else:
    assert comparison=='AUTHORIZED_CROSS_GPU_COMPARISON_RECORDED'
    assert js('engineering_rendezvous_recovery_20261102_v1/'+case+'_TASK6_GPU_COMPARISON.json',js('engineering_resume_3090_20261102_v1/'+case+'_TASK6_GPU_COMPARISON.json'))['status']=='RECORDED_AUTHORIZED_CROSS_GPU_COMPARISON'
 copycheck=js(prefix+'COPY_EQ.json',{})
 if h:
  for st in range(6,h[-1]['stage']+1):
   start=sum(p['stages'][str(t)]['batches']*100 for t in range(6,st))
   assert copycheck['checked'][str(start)]['all_state_exact']
 own=ownauto=0
 for st in range(6,10):
  a=js(prefix+'ADMISSION_S_%d.json'%st)
  if a:
   if st==6:
    assert a['status']=='PASS' and a['optimizer_calls']==4 and all(a[k] for k in ['roundtrip_exact','next_step_exact','restore_initial_exact','baseline_or_legacy_update_exact'])
    short=js(prefix+'SHORT_HISTORY_ADMISSION_S.json');assert short['status']=='PASS' and short['extra_autograd_calls']==8
    own+=4;ownauto+=8
   else:
    assert a['lambda_R']==p['lambda_R'] and a['uniform_reference']=='same_lambda_uniform_support'
    assert a['status']=='PASS_OWN_GPU_ADMISSION' and a['optimizer_calls']==29 and a['extra_autograd_calls']==11
    assert all(a[k] for k in ['q_one_original_S_exact','lambda_zero_original_C_exact','ten_step_C_replay_exact','accepted_and_rejected_full_state_restore','measurement_state_exact','probe_all_state_exact'])
    own+=29;ownauto+=11
    pr=js(prefix+'PROBES_%d.json'%st)
    for t,q in pr['support'].items():
     m=js(prefix+'MEMORY_S_%s.json'%t);ids=set(x['trajectory'] for x in m['rows'])
     assert len(ids)==m['collection_successes'] and q==len(ids)/10 and sorted(ids)==pr['support_trajectories'][t]
 out=js(prefix+'OUTCOMES.json',[]);seen=set()
 for x in out:
  key=(x['stage'],x['task'],x['initial_index']);assert key not in seen;seen.add(key)
  assert x['arm']=='S' and 6<=x['task']<=x['stage']<=9 and 0<=x['initial_index']<20 and x['attempt']==1
  assert 1<=x['steps']<=600 and x['termination']==('success' if x['success'] else 'timeout600') and (x['success'] or x['steps']==600)
 collected=0
 for st in range(6,10):
  m=js(prefix+'MEMORY_S_%d.json'%st)
  if m:
   assert m['training_attempts']==10 and m['count']<=128 and len(m['rows'])==m['count'];collected+=10
 status=js(prefix+'TRAIN_STATUS.json',{})
 current=[row for row in h if row['stage']==status.get('stage')]
 recent=current[-10:];mean=sum(x['seconds'] for x in recent)/len(recent) if recent else None
 counts[case]=dict(epochs=len(h),accepted_updates=acc,physical_optimizer_calls=phys,own_isolated_optimizer_calls=own,own_extra_autograd_calls=ownauto,training_collection_completed_episodes=collected,formal_episodes=len(out),current=status,current_stage_epochs=len(current),recent_epoch_mean_seconds=mean,current_stage_compute_minutes_estimate=(100-len(current))*mean/60 if mean is not None and status.get('phase')=='train' else None,gate_blocks_observed=len(gates),gate_Q_accepts_observed=sum(g['accepted']=='Q' for g in gates))
 epochs+=len(h);accepted+=acc;physical+=phys;admission_calls+=own;admission_autograd+=ownauto;formal+=len(out);collection+=collected
mitigated=js('engineering_rendezvous_recovery_20261102_v1/MITIGATED_FAILURES.json')
assert set(mitigated)=={'L005_43/train_S.exit','engineering_live_parallel_3090_20261102_v1/controller.exit','engineering_live_parallel_3090_20261102_v1/FAILURE.json'}
for n,h in mitigated.items():assert v['light'][n]['sha256']==h
for n,x in v['light'].items():
 if n in mitigated:continue
 if n.endswith('.exit') and x['text'].strip()!='0':failures[n]=x['text']
 if n.endswith('FAILURE.json'):failures[n]=js(n)
if not v['processes'] and v['light'].get('engineering_rendezvous_recovery_20261102_v1/controller.exit',{}).get('text','').strip()!='0':
 failures['missing_controller_without_success']='No own process and no successful controller exit'
if any(x['spare']<0 for x in v['capacity'].values()):failures['capacity']='Reserve or remaining-budget bound violated'
if 'engineering_live_parallel_3090_20261102_v1/WATCHDOG_RELEASE.json' in v['light']:failures['engineering_watchdog_released']='Owner failed before safe boundary; audit original coordinator before any further deployment'
assert v['engineering']['contract_sha256']==json.loads(active.read_text(encoding='utf8'))['engineering_contract']
prior=Path(sys.argv[2]) if len(sys.argv)>2 else None
if prior:
 old=json.loads((prior/'EVIDENCE.json').read_text(encoding='utf8'))
 for n,x in old['light'].items():
  if n.endswith('EPOCHS_S.json'):
   before=json.loads(x['text']);assert js(n,[])[:len(before)]==before
  if n.endswith(('GRADIENTS_S.json','OUTCOMES.json')):
   before=json.loads(x['text']);assert js(n,[])[:len(before)]==before
  if '/GATES_stage' in n:
   assert v['light'][n]['text'].startswith(x['text'])
out=dict(status='ENGINEERING_FAILURE_REQUIRES_REVIEW' if failures else 'PASS_PROGRESS_NOT_FINAL',observation=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),counts=counts,epochs=epochs,accepted_updates=accepted,physical_training_optimizer_calls=physical,own_isolated_optimizer_calls=admission_calls,own_extra_autograd_calls=admission_autograd,training_collection_completed_episodes=collection,formal_episodes=formal,failures=failures,processes=v['processes'],capacity=v['capacity'],gpu=v['gpu_diagnostic'],snapshot_sha256=sha(D/'EVIDENCE.json'),contract=v['contract_sha256'],prefix_checked=str(prior) if prior else None)

eng=v['engineering'];assert eng['diagnostic_optimizer_calls']==35418 and eng['diagnostic_environments']==0
assert 'controller.exit' not in v['light']
for case,cc in counts.items():
 if case=='L005_41' or cc['epochs']<=100:continue
 actual=js('engineering_rendezvous_recovery_20261102_v1/'+case+'_BRANCH_COPY_EQ.json',js('engineering_resume_3090_20261102_v1/'+case+'_BRANCH_COPY_EQ.json'));assert actual['status']=='PASS_OBSERVED_STAGE_STARTS'
 for stage,start in [(7,15700),(8,38400),(9,69300)]:
  if cc['epochs']>(stage-6)*100:assert actual['checked'][str(start)]['full_state_values_exact']
assert 'engineering_resume_3090_20261102_v1/controller.exit' not in v['light']
assert eng['checks_diagnostic_optimizer_calls']==7828 and eng['checks_diagnostic_extra_autograd']==36 and eng['diagnostic_extra_autograd']==94
for case,cc in counts.items():
 if case=='L005_41' or not cc['epochs']:continue
 eq=js('engineering_rendezvous_recovery_20261102_v1/'+case+'_CHECK_EQ.json',js('engineering_checks_3090_20261102_v1/'+case+'_CHECK_EQ.json'));assert eq['status']=='PASS_OBSERVED_CHECKS' and eq['optimizer_calls_added']==eq['environments_added']==0
 for stage,start in [(6,0),(7,15700),(8,38400),(9,69300)]:
  if cc['epochs']>(stage-6)*100:
   assert eq['finite_stage_starts'][str(start)]['exact']
   if stage>6:assert eq['device_stage_starts'][str(start+10)]['exact']
assert eng['abandoned_partial_recorded_optimizer_calls']==140 and eng['abandoned_partial_unrecorded_cost']=='UNKNOWN'
out.update(hardware_migration='USER_AUTHORIZED_RTX3090',abandoned_partial_recorded_optimizer_calls=140,abandoned_partial_unrecorded_cost='UNKNOWN',engineering=eng,engineering_diagnostic_optimizer_calls=35418,engineering_diagnostic_extra_autograd=eng['diagnostic_extra_autograd'])


assert 'engineering_checks_3090_20261102_v1/controller.exit' not in v['light']
assert eng['parallel_diagnostic_optimizer_calls']==18584 and eng['parallel_diagnostic_extra_autograd']==0
pa=js('engineering_parallel_3090_20261102_v1/PARALLEL_ADMISSION.json');assert pa['status']=='PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'
assert pa['optimizer_started']==pa['optimizer_completed']==18584 and pa['environments']==0
events=[];acquired={};released={}
for case in ['L005_42','L005_43','L020_41','L020_42','L020_43']:
 n='engineering_live_parallel_3090_20261102_v1/'+case+'_COLLECTION_LOCK.jsonl'
 for row in v['light'].get(n,{}).get('text','').splitlines():
  x=json.loads(row);key=(case,int(x['stage']));target=acquired if x['event']=='acquired' else released
  assert x['event'] in ['acquired','released'] and key not in target;target[key]=x
  events.append((x['time'],x['event'],key))
  if x['event']=='released':assert key in acquired and x['source']=='pre_import_startup_hook'
active_collection=None
for when,event,key in sorted(events):
 if event=='acquired':assert active_collection is None;active_collection=key
 else:assert active_collection==key;active_collection=None
for key in released:
 case,st=key;assert js(case+'/MEMORY_S_%d.json'%st)['training_attempts']==10
expected={(case,int(st)) for case,stages in js('engineering_live_parallel_3090_20261102_v1/GUARD.json')['expected_collection_stages'].items() for st in stages}
assert set(acquired)<=expected
if formal==1200:assert set(released)==set(acquired)==expected and active_collection is None
root_status=js('STATUS.json',{})
if root_status.get('controller')==json.loads(active.read_text(encoding='utf8'))['controller']:
 running=root_status.get('active',{});assert len(running)<=2
 if len(running)>1:assert all(x['phase']=='train' for x in running.values()) and (set(running)=={'L005_42','L005_43'} or all(k.startswith('L020_') for k in running))
if any(cc['epochs'] for case,cc in counts.items() if case.startswith('L020_')):
 assert sum(counts['L005_'+str(seed)]['formal_episodes'] for seed in [41,42,43])==600
 assert js('L005_EVALUATION_COMPLETE.json')['episodes']==600
out['parallel_collection_lock_audit']=dict(acquisitions=len(acquired),releases=len(released),active=active_collection,nonoverlap=True)


assert 'engineering_parallel_3090_20261102_v1/controller.exit' not in v['light']
if 'engineering_live_parallel_3090_20261102_v1/OWNER_LOST_AFTER_LIVE_COMMIT.json' in v['light']:
 out['status']='ENGINEERING_FAILURE_REQUIRES_REVIEW';out['failures']['live_owner_lost']='Keep frozen waiters; never duplicate current runs'
if 'engineering_live_parallel_3090_20261102_v1/LIVE_COMMITTED.json' in v['light']:
 assert js('engineering_live_parallel_3090_20261102_v1/LIVE_COMMITTED.json')['training_signals']==0
out['parallel_collection_lock_audit']['expected_total']=len(expected)


assert eng['rendezvous_recovery_optimizer_calls']==869 and eng['rendezvous_recovery_extra_autograd']==11 and eng['rendezvous_partial_recorded_optimizer_calls']==400 and eng['total_recorded_abandoned_partial_optimizer_calls']==540
assert eng['rendezvous_partial_unrecorded_cost']=='UNKNOWN' and eng['independent_rendezvous_verified']
out.update(mitigated_failures_preserved=mitigated,rendezvous_partial_recorded_optimizer_calls=400,total_recorded_abandoned_partial_optimizer_calls=540,rendezvous_partial_unrecorded_cost='UNKNOWN')
launches=[json.loads(x) for x in v['light'].get('engineering_rendezvous_recovery_20261102_v1/LAUNCHES.jsonl',{}).get('text','').splitlines()]
for x in launches:
 if x['phase']=='train':
  assert '--standalone' not in x['argv'] and '--rdzv_endpoint=localhost:0' in x['argv'] and '--rdzv_backend=c10d' in x['argv'] and '--rdzv_id='+x['rdzv_id'] in x['argv']
running=js('STATUS.json',{}).get('active',{})
actual=[js('engineering_rendezvous_recovery_20261102_v1/'+case+'_RENDEZVOUS.json') for case,x in running.items() if x['phase']=='train']
actual=[x for x in actual if x is not None]
assert all(x['world_size']==1 for x in actual)
assert len({x['master_port'] for x in actual})==len(actual)
out['rendezvous_audit']=dict(launches=len(launches),observed_training_endpoints=actual,independent_endpoint_arguments=True)

(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
