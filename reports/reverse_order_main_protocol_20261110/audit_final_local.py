"""Final acceptance only after actual remote strict CPU checks and complete outcomes."""
from pathlib import Path
import hashlib,json,sys
from audit_progress import audit,decode,read
d=Path(sys.argv[1]);a=audit(d);e=read(d/'EVIDENCE.json')
assert e['status']=='PASS_REMOTE_REVERSE_CPU_PENDING_LOCAL_ACCEPTANCE'
assert a['controller_exit']=='0' and not a['processes'] and a['environment_held'] is None
assert a['totals']==dict(epochs=4800,accepted=1156800,production_physical=1636500,admission=666,training_collection=240,formal=2400)
obj=decode(e)
events=obj['coordinator/EVENTS.jsonl']
verified_failures=[]
for key,name,expected in (
    ('evaluation_setup_repair_sha256','EVALUATION_SETUP_REPAIR.json','PRE_ROLLOUT_ASSETS_RESTORED'),
    ('evaluation_spawn_repair_sha256','EVALUATION_SPAWN_REPAIR.json','PRE_ROLLOUT_SPAWN_ENTRY_REPAIRED'),
):
    rows=[row for row in events if key in row]
    assert len(rows)<=1
    if rows:
        repair=read(Path(__file__).resolve().parent/name)
        receipt=repair['receipt']
        encoded=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
        assert rows[0][key]==hashlib.sha256(encoded).hexdigest()
        assert rows[0]['pid']==repair['controller'] and receipt['contract']==a['contract']
        assert receipt['status']==expected
        assert receipt['resets']==receipt['steps']==receipt['formal_outcomes']==0
        start=events.index(receipt['failed_events'][0])
        assert events[start:start+4]==receipt['failed_events']
        assert events[start+2]['kind']=='environment_released' and events[start+2]['success'] is False
        for filename,digest in receipt.get('completed_hashes',{}).items():
            assert e['light'][receipt['completed_case']+'/'+filename]['sha256']==digest
        verified_failures.append(events[start+2])
assert a['closed_environment_intervals']==36+len(verified_failures) and len(e['cpu'])==12
formal_releases=[row for row in events if row['kind']=='environment_released' and row['purpose']=='formal']
assert [row for row in formal_releases if row['success'] is False]==verified_failures
assert len(formal_releases)==12+len(verified_failures) and sum(row['success'] is True for row in formal_releases)==12
dispatch_repairs=[row for row in events if 'evaluation_dispatch_repair_sha256' in row]
assert len(dispatch_repairs)<=1
if dispatch_repairs:
    repair=read(Path(__file__).resolve().parent/'EVALUATION_DISPATCH_REPAIR.json')
    receipt=repair['receipt']
    encoded=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
    assert dispatch_repairs[0]['evaluation_dispatch_repair_sha256']==hashlib.sha256(encoded).hexdigest()
    assert dispatch_repairs[0]['pid']==repair['controller'] and receipt['contract']==a['contract']
    assert receipt['status']=='PRE_DISPATCH_METADATA_REPAIRED'
    assert receipt['failed_dispatcher'] in events and receipt['failed_launches']==0
    assert not any(row['kind']=='environment_acquired' and row['pid']==receipt['failed_dispatcher']['pid'] for row in events)
for case,x in e['cpu'].items():
    assert x['status']=='PASS_REVERSE_CPU_CASE' and x['initial_physical_bytes']
    assert len(x['checkpoints'])==4 and all(v['strict_load'] and v['strict_keys']==100 and v['cpu_serialization_and_forward_exact'] for v in x['checkpoints'].values())
    assert x['recovery']['model_optimizer_scheduler_grads_modes_requires_rng_step']
    assert x['new_optimizer']==x['new_environments']==0
    assert len(x['caches'])==(4 if x['method']=='DEBR' else 0)
    assert x['evaluation_protocol']=='COMMON_DEBR_EVALUATION_20261111_V1'
    assert x['noise_definition']=='common_full_DEBR_per_trajectory_Normal'
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        m=obj['%s_MLR_%d/NORMALIZED_OUTCOMES.json'%(suite,seed)]
        s=obj['%s_DEBR_%d/OUTCOMES.json'%(suite,seed)]
        assert len(m)==len(s)==200
        for x,y in zip(m,s):
            assert all(x[k]==y[k] for k in ['stage','task','original_task','initial_index','initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt'])
result=dict(status='PASS_COMPLETE_REVERSE_LOCAL_ACCEPTANCE',observed=e['observed'],contract=a['contract'],runs=12,checkpoints=48,recoveries=12,caches=24,totals=a['totals'],pre_rollout_setup_failures=len(verified_failures),actual_cpu=True,new_final_optimizer=0,new_final_environment=0,evidence_sha256=hashlib.sha256((d/'EVIDENCE.json').read_bytes()).hexdigest())
(d/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
