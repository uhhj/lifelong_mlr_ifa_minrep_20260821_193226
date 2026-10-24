"""Independent local metadata, population, admission and paired-record audit."""
import sys,json,hashlib,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'));light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];p=D/'light'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x['text'],encoding='utf8');light[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
pop={6:997,7:1388,8:1848,9:3365};valid={6:917,7:1286,8:1723,9:3218};olds={6:0,7:477,8:899,9:1329};counts={};rates={};admissions={};teachers={}
parent=R.parent/'20261022_align_object_extension_goal_conditional/final_GOAL_41_20261023_052225/current'
for t in range(6,10):
 p=parent/('ra_coreset/dataset_manifests/task%d.json'%t);rows=json.loads(p.read_text(encoding='utf8'))['rows']
 assert len(rows)==pop[t] and sum(r['bc_valid'] for r in rows)==valid[t] and sum(r['bc_valid'] and r['task']<t for r in rows)==olds[t]
 for arm in 'BCNV':
  key='TEACHER_%s_%d.json'%(arm,t)
  if key in light:
   m=light[key];assert t>6 and m['count']==olds[t] and m['manifest_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
   expected={i:r for i,r in enumerate(rows) if r['task']<t and r['bc_valid']};assert {r['index'] for r in m['rows']}==set(expected)
   costs=[]
   for row in m['rows']:
    assert row['sample_sha256']==expected[row['index']]['sha256'] and row['task']==expected[row['index']]['task'];costs.append(row['teacher_cost'])
    assert abs(row['weight']-m['tau']/(m['tau']+row['teacher_cost']))<1e-7
   assert abs(m['tau']-max(1e-8,float(np.median(costs))))<1e-14
   teachers[key]=dict(count=m['count'],tau=m['tau'],cache_sha256=m['cache_sha256'],source_sha256=m['source_sha256'])
for arm in 'BCNV':
 rows=light.get('EPOCHS_'+arm+'.json',[]);steps=0
 for i,row in enumerate(rows):
  t=6+i//100;ep=i%100;b=(pop[t]+9)//10;steps+=b
  assert (row['stage'],row['epoch'],row['batches'],row['steps'])==(t,ep,b,steps)
  assert all(row['totals'][k]==val for k,val in [('valid',valid[t]),('invalid',pop[t]-valid[t]),('old_eligible',olds[t])])
 counts[arm]=dict(epochs=len(rows),steps=steps,by_stage={str(t):sum(r['stage']==t for r in rows) for t in range(6,10)})
 rates[arm]={str(t):float(np.mean([r['seconds'] for r in rows if r['stage']==t][-10:])) for t in range(6,10) if any(r['stage']==t for r in rows)}
 for t in range(6,10):
  key='ADMISSION_%s_%d.json'%(arm,t)
  if key in light:
   a=light[key];assert a['status']=='PASS' and a['roundtrip_exact'] and a['next_step_exact'] and a['restore_initial_exact'] and a['strict_keys']==100 and a['environments']==0
   assert a['optimizer_calls']==(4 if t==6 else 5 if arm in 'BC' else 4)
   if t>6:assert a['stateful_gradient_measurement_exact']
   if t==6 or arm in 'BC':assert a['baseline_or_legacy_update_exact']
   admissions[key]=a['optimizer_calls']
 if 'INITIAL_'+arm+'.json' in light:
  a=light['INITIAL_'+arm+'.json'];parent_initial=json.loads((parent/'ra_coreset/INITIAL_STATE.json').read_text(encoding='utf8'));assert a['tensors']==parent_initial['tensors'] and a['rng']==parent_initial['rng']
 for other in 'BCNV':
  for x,y in zip(rows,light.get('EPOCHS_'+other+'.json',[])):assert x['sampler_sha256']==y['sampler_sha256']
records=light.get('OUTCOMES.json',[]);seen=set();initials={}
for r in records:
 key=(r['arm'],r['stage'],r['task'],r['initial_index']);assert key not in seen;seen.add(key)
 assert r['arm'] in 'BCNV' and 6<=r['task']<=r['stage']<=9 and 0<=r['initial_index']<20
 assert r['attempt']==1 and r['env_seed']==41+r['initial_index']//10 and 0<r['steps']<=600 and (r['success'] or r['steps']==600)
 noise=np.random.RandomState(20261024+41000+r['task']*100+r['initial_index']).standard_normal((600,7)).astype(np.float32);assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256']
 ik=(r['task'],r['initial_index']);binding=(r['initial_file_sha256'],r['initial_sha256'],r['noise_sha256']);assert initials.setdefault(ik,binding)==binding
 stage=light.get('STAGE_%s_%d.json'%(r['arm'],r['stage']));assert stage and stage['sha256']==r['cp_sha256']
failure=light.get('FAILURE.json')
if 'REPAIR_CONTINUATION.json' in light and 'RESTORE_ADMISSION.json' in light:
 b=light['REPAIR_CONTINUATION.json'];r=light['RESTORE_ADMISSION.json'];assert r['all_fields_exact'] and r['completed_epochs_repeated']==0 and r['recovery_sha256']==b['recovery_sha256'] and r['previous_contract']==b['previous_contract']
 original=json.loads((R/'check_readonly_20261025_000623/light/EPOCHS_B.json').read_text(encoding='utf8'));assert light['EPOCHS_B.json'][:100]==original
if not failure and light.get('controller.exit')!='0' and not v.get('processes',[]):failure={'reason':'No successful controller exit and no own process'}
controller_phase=light.get('STATUS.json',{})
phase=controller_phase if controller_phase.get('phase') in ('evaluate','complete') else light.get('TRAIN_STATUS.json',controller_phase)
out=dict(status='FAILURE_REQUIRES_INSPECTION' if failure else 'PASS_PROGRESS_NOT_FINAL',observation=v['observed'],contract=v.get('contract_sha256',v.get('contract')),counts=counts,rates_last10=rates,admissions=admissions,isolated_optimizer_calls=sum(admissions.values()),teacher_bindings=teachers,formal_episodes=len(records),no_partial_performance_claim=True,failure=failure,phase=phase,processes=v.get('processes',[]))
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
