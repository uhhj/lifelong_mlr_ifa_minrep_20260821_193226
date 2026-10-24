import sys,json,hashlib,statistics,math
from pathlib import Path
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']
 if n.endswith('.json'):light[n]=json.loads(x['text'])
teacher=light['TEACHER.json'];weights=light['WEIGHTS.json'];assert teacher['reused_readonly'] and teacher['new_forward_batches']==0 and len(weights['rows'])==teacher['count']==430
assert teacher['weights_sha256']==v['light']['WEIGHTS.json']['sha256'];tau=max(1e-8,statistics.median(r['teacher_cost'] for r in weights['rows']));assert tau==weights['tau']==teacher['tau']
assert len({r['index'] for r in weights['rows']})==430 and len({r['sample_sha256'] for r in weights['rows']})==430
assert all(math.isfinite(r['weight']) and abs(r['weight']-tau/(tau+r['teacher_cost']))<1e-7 for r in weights['rows'])
admit={a:light['ADMISSION_'+a+'.json'] for a in 'BCNV'}
for arm,a in admit.items():
 assert a['status']=='PASS' and a['contract']==v['contract_sha256'] and a['roundtrip_exact'] and a['next_step_exact'] and a['gradient_measurement_update_exact'] and a['shared_storage_alias_exact']
 assert a['parameters']==21633611 and a['strict_keys']==100 and a['shared_storage_serialization_bytes']<350000000 and a['environments']==0
 assert a['optimizer_calls']==(5 if arm in 'BC' else 4) and (arm not in 'BC' or a['legacy_control_update_exact'])
 assert a['synthetic']['uploaded_prototype_loss_gradient_exact'] and a['synthetic']['teacher_no_grad']
out=dict(status='PASS_OWN_ADMISSION_AND_FROZEN_WEIGHT_BINDING_NOT_FINAL',observation=v['observed'],contract=v['contract_sha256'],own_optimizer_calls=sum(a['optimizer_calls'] for a in admit.values()),own_admission_extra_gradient_autograd_calls=16,formal_environments=0,teacher_cache_readonly=True,teacher_tau=tau,weight_min=min(r['weight'] for r in weights['rows']),weight_max=max(r['weight'] for r in weights['rows']),weight_mean=statistics.mean(r['weight'] for r in weights['rows']),weights_sha256=teacher['weights_sha256'],snapshot_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest())
(D/'ADMISSION_AND_WEIGHT_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
