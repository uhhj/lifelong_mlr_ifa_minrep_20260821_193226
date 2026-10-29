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
for seed in (41,42,43):
 prefix='seed%d/'%seed;p=json.loads((R/prefix/'PARENT_BINDING.json').read_text(encoding='utf8'))
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
 if h:
  for stage in range(6,h[-1]['stage']+1):assert prefix+'ADMISSION_S_%d.json'%stage in v['light']
 own=ownauto=0
 for st in range(6,10):
  a=js(prefix+'ADMISSION_S_%d.json'%st)
  if a:
   if st==6:
    assert a['status']=='PASS' and a['optimizer_calls']==4 and all(a[k] for k in ['roundtrip_exact','next_step_exact','restore_initial_exact','baseline_or_legacy_update_exact'])
    short=js(prefix+'SHORT_HISTORY_ADMISSION_S.json');assert short['status']=='PASS' and short['extra_autograd_calls']==8
    own+=4;ownauto+=8
   else:
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
 counts[str(seed)]=dict(epochs=len(h),accepted_updates=acc,physical_optimizer_calls=phys,own_isolated_optimizer_calls=own,own_extra_autograd_calls=ownauto,training_collection_completed_episodes=collected,formal_episodes=len(out),current=status,current_stage_epochs=len(current),recent_epoch_mean_seconds=mean,current_stage_compute_minutes_estimate=(100-len(current))*mean/60 if mean is not None and status.get('phase')=='train' else None,gate_blocks_observed=len(gates),gate_Q_accepts_observed=sum(g['accepted']=='Q' for g in gates))
 epochs+=len(h);accepted+=acc;physical+=phys;admission_calls+=own;admission_autograd+=ownauto;formal+=len(out);collection+=collected
for n,x in v['light'].items():
 if n.endswith('.exit') and x['text'].strip()!='0':failures[n]=x['text']
 if n.endswith('FAILURE.json'):failures[n]=js(n)
if not v['processes'] and v['light'].get('engineering_copy_20261029_v1/controller.exit',{}).get('text','').strip()!='0':
 failures['missing_controller_without_success']='No own process and no successful controller exit'
if any(x['spare']<0 for x in v['capacity'].values()):failures['capacity']='Reserve or remaining-budget bound violated'
prior=Path(sys.argv[2]) if len(sys.argv)>2 else None
if prior:
 old=json.loads((prior/'EVIDENCE.json').read_text(encoding='utf8'))
 for n,x in old['light'].items():
  if n.endswith('EPOCHS_S.json'):
   before=json.loads(x['text']);assert js(n,[])[:len(before)]==before
  if '/GATES_stage' in n:
   assert v['light'][n]['text'].startswith(x['text'])
eng=js('engineering_copy_20261029_v1/VALIDATION.json')
assert eng['status']=='PASS_FULL_TRAINING_PATH_EXACT' and eng['optimizer_calls_started']==eng['optimizer_calls_completed']==3914
assert eng['environments']==eng['accepted_scientific_epochs']==0 and eng['completed_seed41_artifacts_unchanged']
assert len(eng['stages'])==4 and [x['batches'] for x in eng['stages']]==[157,227,309,362]
assert v['engineering_files']==5 and v['completion_source']=='engineering_copy_20261029_v1/controller.exit'
engineering=dict(contract=v['engineering_contract_sha256'],diagnostic_optimizer_calls=3914,diagnostic_extra_autograd_calls=eng['extra_autograd_calls'],diagnostic_environments=0,completion_source=v['completion_source'],validation_scope=eng['scope'])
out=dict(status='ENGINEERING_FAILURE_REQUIRES_REVIEW' if failures else 'PASS_PROGRESS_NOT_FINAL',observation=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),engineering=engineering,counts=counts,epochs=epochs,accepted_updates=accepted,physical_training_optimizer_calls=physical,own_isolated_optimizer_calls=admission_calls,own_extra_autograd_calls=admission_autograd,training_collection_completed_episodes=collection,formal_episodes=formal,failures=failures,processes=v['processes'],capacity=v['capacity'],gpu=v['gpu_diagnostic'],snapshot_sha256=sha(D/'EVIDENCE.json'),contract=v['contract_sha256'],prefix_checked=str(prior) if prior else None)
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
