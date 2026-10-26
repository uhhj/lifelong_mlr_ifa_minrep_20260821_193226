"""Local recomputation of epochs, fixed caches, collection budget and paired records."""
import sys,json,hashlib,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);seed=int(sys.argv[2]);binding=json.loads((R/('seed%d'%seed)/'PARENT_BINDING.json').read_text(encoding='utf8'));js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');L={}
for n,x in v['light'].items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];p=D/'light'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x['text'],encoding='utf8');L[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
pop={int(t):x['population'] for t,x in binding['stages'].items()};valid={int(t):x['valid'] for t,x in binding['stages'].items()};old={int(t):x['old_valid'] for t,x in binding['stages'].items()}
split=binding['initial_split'];counts={};rates={};admissions={};memory={};collection=[]

def outcomes(rows,formal):
 seen=set()
 for r in rows:
  arm,t,i=r['arm'],r['task'],r['initial_index'];stage=r['stage'] if formal else t
  key=(arm,stage,t,i);assert key not in seen;seen.add(key)
  assert arm in 'S' and 6<=t<=stage<=9 and i in (range(20) if formal else range(20,30))
  assert r['attempt']==1 and 0<r['steps']<=600 and r['termination']==('success' if r['success'] else 'timeout600') and (r['success'] or r['steps']==600)
  assert r['env_seed']==(seed+i//10 if formal else seed+100)
  assert split[str(t)]['file_sha256']==r['initial_file_sha256'] and split[str(t)]['evaluation' if formal else 'training'][str(i)]==r['initial_sha256']
  noise=np.random.RandomState((20261024 if formal else 20271025)+seed*1000+t*100+i).standard_normal((600,7)).astype(np.float32)
  assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256']
  assert L['STAGE_%s_%d.json'%(arm,stage)]['sha256']==r['cp_sha256']
for arm in 'S':
 hist=L.get('EPOCHS_'+arm+'.json',[]);steps=0
 for i,r in enumerate(hist):
  t=6+i//100;ep=i%100;b=(pop[t]+9)//10;steps+=b
  assert (r['stage'],r['epoch'],r['batches'],r['steps'])==(t,ep,b,steps)
  assert all(r['totals'][k]==x for k,x in [('valid',valid[t]),('invalid',pop[t]-valid[t]),('old_eligible',old[t]),('anchor_slots',4*b if t>6 else 0)])
  assert r['totals']['anchor_slots']==r['totals']['anchor_actual']+r['totals']['anchor_empty']
 counts[arm]=dict(epochs=len(hist),steps=steps,by_stage={str(t):sum(r['stage']==t for r in hist) for t in range(6,10)})
 rates[arm]={str(t):float(np.mean([r['seconds'] for r in hist if r['stage']==t][-10:])) for t in range(6,10) if any(r['stage']==t for r in hist)}
 for other in 'S':
  for x,y in zip(hist,L.get('EPOCHS_'+other+'.json',[])):assert x['sampler_sha256']==y['sampler_sha256']
 for t in range(6,10):
  key='%s_%d'%(arm,t);a=L.get('ADMISSION_'+key+'.json')
  if a:
   assert a['status']=='PASS' and all(a[k] for k in ['roundtrip_exact','next_step_exact','restore_initial_exact','baseline_or_legacy_update_exact','stateless_sampling_replay_exact'])
   assert a['strict_keys']==100 and a['environments']==0 and a['optimizer_calls']==(4 if t==6 else 6)
   if t>6:assert a['stateful_gradient_measurement_exact']
   admissions[key]=a['optimizer_calls']
  inp=L.get('TRAIN_INPUT_'+key+'.json')
  if inp:
   assert inp['manifest_sha256']==binding['stages'][str(t)]['manifest_sha256']
   assert set(inp['memory'])=={str(j) for j in range(6,t)}
   for j,binding in inp['memory'].items():
    m=L['MEMORY_%s_%s.json'%(arm,j)];assert binding==dict(cache_sha256=m['cache_sha256'],source_sha256=m['source_sha256'],count=m['count'])
  rr=L.get('COLLECTION_OUTCOMES_'+key+'.json')
  if rr:
   assert len(rr)==10 and [r['initial_index'] for r in rr]==list(range(20,30));outcomes(rr,False);collection+=rr
  m=L.get('MEMORY_'+key+'.json')
  if m:
   assert rr and m['count']==len(m['rows'])<=128 and m['training_attempts']==10 and m['source_sha256']==L['STAGE_'+key+'.json']['sha256']
   assert m['collection_outcomes_sha256']==hashlib.sha256(v['light']['COLLECTION_OUTCOMES_'+key+'.json']['text'].encode()).hexdigest()
   assert m['collection_successes']==sum(r['success'] for r in rr)
   ids=set();traj={}
   for row in m['rows']:
    assert row['task']==t and 1<=row['length']<=8
    if arm=='D':
     idx=row['source_index'];source=manifests[t][idx];assert source['task']==t and source['bc_valid'] and source['sha256']==row['source_sha256'] and source['demo']==row['demo'] and source['u']==row['u'];identity=idx
    else:
     j=row['trajectory'];assert row['initial_index']==rr[j]['initial_index'] and 0<=row['step']<rr[j]['steps'];assert row['length']==min(row['step']+1,8)
     assert arm=='O' or rr[j]['success'];traj[j]=traj.get(j,0)+1;assert traj[j]<=13 if j<8 else traj[j]<=12;identity=(j,row['step'])
    assert identity not in ids;ids.add(identity)
   memory[key]=dict(count=m['count'],successes=m['collection_successes'],source_sha256=m['source_sha256'],cache_sha256=m['cache_sha256'])
   ca=L['COLLECTION_ADMISSION_'+key+'.json'];assert ca['status']=='PASS' and ca['updates']==ca['environments']==0 and ca['cp_sha256']==m['source_sha256']
  parity=L.get('COLLECTION_STATE_PARITY_'+key+'.json')
  if parity:assert parity['status']=='PASS' and parity['parent_all_state_exact'] and m and parity['memory_sha256']==m['cache_sha256']
 assert sum(m['count'] for key,m in memory.items() if key.startswith(arm+'_'))<=512
 if 'INITIAL_'+arm+'.json' in L:
  a=L['INITIAL_'+arm+'.json'];assert a['status']=='PASS_OFFICIAL_INITIALIZATION' and a['parent_initial_sha256']==binding['files'][binding['output']+'/ra_coreset/INITIAL_STATE.json'] and a['official_cp_sha256']==binding['official_initialization']['sha256']
 sh=L.get('SHORT_HISTORY_ADMISSION_'+arm+'.json')
 if sh:assert sh['status']=='PASS' and sh['all_state_restored'] and sh['eight_frame_legacy_exact'] and [x['length'] for x in sh['lengths']]==list(range(1,9)) and sh['extra_autograd_calls']==8 and sh['optimizer_calls']==sh['environments']==0
records=L.get('OUTCOMES.json',[]);outcomes(records,True)
failure=L.get('FAILURE.json')
if not failure and L.get('controller.exit')!='0' and not v.get('processes',[]):failure=dict(reason='No success exit and no own process')
status=L.get('STATUS.json',{});phase=status if status.get('phase') in ['evaluate','complete'] else L.get('TRAIN_STATUS.json',status)
out=dict(status='FAILURE_REQUIRES_INSPECTION' if failure else 'PASS_PROGRESS_NOT_FINAL',observation=v['observed'],contract=v['contract_sha256'],counts=counts,rates_last10=rates,admissions=admissions,isolated_optimizer_calls=sum(admissions.values()),memory=memory,training_collection_episodes=len(collection),formal_episodes=len(records),failure=failure,phase=phase,processes=v.get('processes',[]),no_partial_performance_claim=True)
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
