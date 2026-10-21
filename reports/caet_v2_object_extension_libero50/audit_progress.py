import json,hashlib,sys,datetime
from pathlib import Path
p=Path(sys.argv[1]);raw=(p/'SNAPSHOT.json').read_bytes();d=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==json.loads((p/'RECEIPT.json').read_text())['sha256']
seed=int(sys.argv[2]);assert seed in (41,43)
W='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021/object_seed%d'%seed;O='/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed%d_20261021'%seed;f=d['files']
assert d['frozen_hashes'] and all(d['frozen_hashes'].values())
gate=f[O+'/GPU_PREFLIGHT.json'];assert gate['status']=='PASS' and gate['optimizer_calls']==8 and gate['formal_steps']==gate['environment_steps']==0
assert gate['zero_aux_native_update_exact'] and gate['full_state_epoch_recovery_exact'] and gate['actual_index_wrapper_order_exact']
r=gate['retention'];assert r['harmful']>r['bound']+r['epsilon'] and r['after_response']<r['harmful'] and 0<r['mu']<=1 and r['gradient_norm2']>0
for x in gate['plasticity']:assert (x['dot']<0)==(x['sign']<0) and x['bc_after']<x['bc_before']
assert f[W+'/admission/gpu_preflight.exit'].strip()=='0'
assert f[O+'/LAUNCH_ACCEPTANCE.json']['status']=='PASS'
h=f.get(O+'/ra_coreset/epochs.json',[]);steps=0
for k,row in enumerate(h):
 assert row['task']==6+k//100 and row['epoch']==k%100
 assert row['batches']=={6:157,7:227,8:309,9:362}[row['task']]
 steps+=row['batches'];assert row['steps']==steps
for t,a in d.get('auxiliary',{}).items():
 assert a['accounting_pass']
 if t in ('6','7'):assert a['active']==a['transfer_applied']==a['retention_active']==0
parity=f.get(O+'/TASK6_TENSOR_PARITY.json');assert parity is None or parity['equal']
assert len(d['formal'])<=200
counts={}
for row in d['formal']:
 key=(row['stage'],row['task'],row['trial']);assert key not in counts;counts[key]=row['success']
 assert row['seed']==seed and row['stage'] in range(6,10) and 6<=row['task']<=row['stage'] and row['trial'] in range(20)
events={}
for row in d['attempts']:events[row['event']]=events.get(row['event'],0)+1
assert events.get('vector_attempt_start',0)<=20 and events.get('vector_attempt_complete',0)<=events.get('vector_attempt_start',0)
assert not any('error' in k or 'exception' in k for k in events)
out=dict(status='PARTIAL_PROGRESS_PASS',observed_at=datetime.datetime.fromtimestamp(d['observed_unix']).astimezone().isoformat(),epochs=len(h),main_optimizer_steps=steps,formal_records=len(d['formal']),gate1='PASS',gate2='PASS',native_off_exact=True,retention_functionality=r,plasticity=gate['plasticity'],auxiliary=d.get('auxiliary',{}),snapshot_inflight_auxiliary=d.get('partial_aux',[]),task6_parity=None if parity is None else parity['equal'],frozen_hashes=len(d['frozen_hashes']),workers=len(d['processes']),gpu=d['gpu'],disk_free=d['disk'],not_final_acceptance=True)
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
