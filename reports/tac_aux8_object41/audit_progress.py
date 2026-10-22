"""Local scalar recomputation; partial observation is not final acceptance."""
import sys,json,hashlib,datetime,math
from pathlib import Path
p=Path(sys.argv[1]);raw=(p/'SNAPSHOT.json').read_bytes();d=json.loads(raw)
assert not d['hashes']['failures'] and d['hashes']['count']==691
v=d['runs']['TAC_AUX8_OBJECT_41'];g=v['files']['GPU_PREFLIGHT.json'];assert g and g['status']=='PASS'
assert g['optimizer_calls']==g['main_optimizer_calls']+g['auxiliary_head_optimizer_calls']==15
assert g['formal_steps']==g['environment_steps']==0
for k in ('zero_aux_native_update_exact','full_state_epoch_recovery_exact','active_aux_recovery_exact','auxiliary_not_in_deployment_state_dict'):assert g[k]
assert g['main_policy_keys']==100 and g['auxiliary_gradient_route']['deployment_head_gradients_absent']
assert 0<g['auxiliary_gradient_route']['temporal_norm2']<math.inf
rr=g['retention'];assert rr['harmful']>rr['bound']+.05 and rr['after_response']<rr['harmful'] and 0<rr['mu']<=1
assert all(x['after']<x['before'] for x in g['plasticity'])
h=v['epochs'];step=0;seen=[];batch={6:157,7:227,8:309,9:362}
for row in h:
 t,e=row['task'],row['epoch'];seen.append((t,e));assert row['batches']==batch[t];step+=row['batches'];assert row['steps']==step
assert seen==[(t,e) for t in (6,7,8,9) for e in range(100)][:len(h)]
for t,x in v['auxiliary'].items():
 assert x['head_updates']<=x['records']
 assert x['transfer']<=x['records'] and x['retention']<=x['records']
 if int(t) in (6,7):assert x['transfer']==x['retention']==0
 if x['last_head']:assert x['last_head']['head_steps']>=x['head_updates'] and math.isfinite(x['last_head']['nll'])
assert all(str(x).strip()=='0' for x in v['exits'].values())
parity=v['files']['TASK6_TENSOR_PARITY.json']
if parity:assert parity['equal']
receipt=dict(status='PARTIAL_PROGRESS_PASS',observed_at=datetime.datetime.fromtimestamp(d['unix']).astimezone().isoformat(),snapshot_sha256=hashlib.sha256(raw).hexdigest(),hash_count=d['hashes']['count'],complete_epochs=len(h),main_steps=step,head_updates=sum(x['head_updates'] for x in v['auxiliary'].values()),head_exposures=sum(x['head_exposures'] for x in v['auxiliary'].values()),formal_records=v['formal_records'],task6_final_tensor_parity=None if not parity else parity['equal'],gpu_admission=g,not_final_acceptance=True)
(p/'LOCAL_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
