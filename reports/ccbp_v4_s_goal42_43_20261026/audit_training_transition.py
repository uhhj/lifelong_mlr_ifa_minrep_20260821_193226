"""Read-only binding of finished training and new seed's own admission, not final CP acceptance."""
import json,sys,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
active=js(R/'ACTIVE_ROOT.json');assert v['contract_sha256']==active['contract'] and v['children']==active['children'] and v['source_files']==active['frozen_files']
out={};skew=[]
for seed in (42,43):
 prefix=f'seed{seed}/';L={n[len(prefix):]:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items() if n.startswith(prefix)}
 hist=L.get('EPOCHS_S.json',[]);st=L.get('TRAIN_STATUS.json',{});grads=L.get('GRADIENTS_S.json',[])
 if hist and st.get('phase')=='train' and st['steps']!=hist[-1]['steps']:skew.append(dict(seed=seed,epochs_metadata_steps=hist[-1]['steps'],status_steps=st['steps'],reason='Files sampled separately during natural progress; completed EPOCHS ledger is the count source.'))
 completed=len(hist)==400;summary=L.get('TRAIN_S.json');seconds=sum(x['seconds'] for x in hist)
 points={(x['stage'],x['epoch']) for x in grads};expected={(t,ep) for t in range(7,10) for ep in (0,9,49,99)};assert len(points)==len(grads) and points<=expected
 for g in grads:
  assert g['extra_autograd_calls']==6
  for term in ['bc','ifa_weighted','expert_weighted','expert_route_weighted','retain_weighted','retain_route_weighted']:
   assert all(math.isfinite(g[term][k]) for k in ['loss','l2']) and g[term]['l2']>=0
 if completed:
  assert L['train_S.exit']=='0' and summary['status']=='PASS' and summary['epochs']==400 and summary['steps']==76100 and points==expected and summary['gradient_autograd_calls']==72
  assert math.isclose(seconds,summary['summed_epoch_seconds'],rel_tol=1e-12)
  assert summary['checkpoints']=={str(t):L[f'STAGE_S_{t}.json']['sha256'] for t in range(6,10)}
 caches={};collect_times=[]
 for t in range(6,10):
  m=L.get(f'MEMORY_S_{t}.json')
  if not m:continue
  rows=L[f'COLLECTION_OUTCOMES_S_{t}.json'];expected_count=sum(min(x['steps'],13 if j<8 else 12) for j,x in enumerate(rows) if x['success']);assert m['count']==expected_count
  assert L[f'collect_S_{t}.exit']=='0' and L[f'COLLECTION_STATE_PARITY_S_{t}.json']['parent_all_state_exact']
  caches[str(t)]=dict(count=m['count'],expected_success_only_quota_count=expected_count,source_sha256=m['source_sha256'],cache_sha256=m['cache_sha256'])
 events=[json.loads(x) for x in L.get('COLLECTION_EVENTS.jsonl','').splitlines()];starts={};ends={}
 for x in events:
  assert x['kind'] in ['start','complete'];dst=starts if x['kind']=='start' else ends;k=(x['arm'],x['task']);assert k not in dst;dst[k]=x
  if x['kind']=='complete':assert k in starts and x['attempts']==10 and x['time']>=starts[k]['time'];collect_times.append(x['seconds'])
 if completed:assert set(starts)==set(ends)=={('S',t) for t in range(6,10)} and len(caches)==4
 for t in range(6,10):
  ad=L.get(f'ADMISSION_S_{t}.json')
  if ad:assert ad['contract']==v['children'][str(seed)]
 sh=L.get('SHORT_HISTORY_ADMISSION_S.json')
 if sh:assert sh['status']=='PASS' and sh['policy_calls']==17 and sh['extra_autograd_calls']==8 and sh['environments']==sh['optimizer_calls']==0
 out[str(seed)]=dict(complete_training=completed,full_epochs=len(hist),main_steps=hist[-1]['steps'] if hist else 0,epoch_seconds=seconds,epoch_seconds_by_stage={str(t):sum(x['seconds'] for x in hist if x['stage']==t) for t in range(6,10)},collection_seconds=sum(collect_times),completed_collection_jobs=len(ends),caches=caches,gradient_measurement_autograd=sum(x['extra_autograd_calls'] for x in grads),own_short_history_admission=sh,task6_historical_parity=L.get('TASK6_PARITY_S.json'))
one=out['42'];two=out['43'];n8=a['seeds']['43']['counts']['S']['by_stage']['8'];rate8=a['seeds']['43']['rates_last10']['S'].get('8')
eta=None
if one['complete_training'] and rate8 and n8<100:
 eta=dict(current_stage8_remaining_minutes=(100-n8)*rate8/60,reference_seed42_stage9_minutes=one['epoch_seconds_by_stage']['9']/60,remaining_compute_reference_minutes=((100-n8)*rate8+one['epoch_seconds_by_stage']['9'])/60,scope='seed43 remaining whole training; own stage9 unmeasured; collection/save/admission extra; not final evaluation ETA')
result=dict(status='PASS_PROGRESS_NOT_FINAL',observation=v['observed'],contract=v['contract_sha256'],seeds=out,metadata_skew=skew,eta=eta,progress_audit_sha256=sha(D/'LOCAL_AUDIT.json'),no_server_or_environment_calls=True,final_CPU_checkpoint_and_cache_replay='PENDING_ALL_RUNS_COMPLETE')
(D/'TRAINING_TRANSITION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
