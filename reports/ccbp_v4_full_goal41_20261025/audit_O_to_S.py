"""Local metadata audit of O completion, S admission and conservative ETA."""
import hashlib
import json
import sys
from pathlib import Path

snap, cap = map(Path, sys.argv[1:3])
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
v = read(snap/'EVIDENCE.json'); light = v['light']
load = lambda n: json.loads(light[n]['text'])
for n,x in light.items():
    assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'],n
audit = read(snap/'LOCAL_AUDIT.json')
assert audit['status']=='PASS_PROGRESS_NOT_FINAL' and audit['phase']['arm']=='S'
assert v['source_files']==89 and v['contract_sha256']=='ef4f4b759632c7c81ebc5ca8da87011e1d4bad4fea48a205b9b30b7c2da1a54a'
assert light['train_O.exit']['text'].strip()=='0'
train=load('TRAIN_O.json');hist=load('EPOCHS_O.json')
assert train['status']=='PASS' and train['epochs']==len(hist)==400 and train['steps']==76100
assert abs(sum(x['seconds'] for x in hist)-train['summed_epoch_seconds'])<1e-8
assert train['gradient_autograd_calls']==72 and train['training_collection_episodes']==40
for t in range(6,10):
    assert light['collect_O_%d.exit'%t]['text'].strip()=='0'
    assert load('COLLECTION_STATE_PARITY_O_%d.json'%t)['parent_all_state_exact']
    assert train['checkpoints'][str(t)]==load('STAGE_O_%d.json'%t)['sha256']
parity=load('TASK6_PARITY_O.json')
assert parity['same_device_D_equal'] and parity['actual_sha256']==train['checkpoints']['6']
for arm in 'OS':
    s=load('SHORT_HISTORY_ADMISSION_%s.json'%arm)
    assert s['status']=='PASS' and s['eight_frame_legacy_exact'] and s['all_state_restored']
    assert [r['length'] for r in s['lengths']]==list(range(1,9))
    assert all(r['finite_gradient'] and r['actual_slots']==4 for r in s['lengths'])
    assert s['extra_autograd_calls']==8 and s['policy_calls']==17 and s['optimizer_calls']==s['environments']==0
events=[json.loads(s) for s in light['COLLECTION_EVENTS.jsonl']['text'].splitlines() if s]
done=[e for e in events if e['kind']=='complete' and e['arm']=='O']
assert len(done)==4 and sum(e['attempts'] for e in done)==40
remain6=(100-audit['counts']['S']['by_stage']['6'])*audit['rates_last10']['S']['6']/60
reference=sum(x['seconds'] for x in hist if x['stage']>6)/60
collect=sum(x['seconds'] for x in done)/60
out=dict(status='PASS_PROGRESS_NOT_FINAL',observation=v['observed'],evidence_sha256=sha(snap/'EVIDENCE.json'),
    O_natural_exit0=True,O_full_parent_state_parity=True,O_Task6_tensor_equal_D=True,
    O_epoch_minutes=train['summed_epoch_seconds']/60,O_collection_minutes=collect,
    OS_short_history_actual_PASS=True,OS_extra_autograd_calls=16,OS_extra_policy_calls=34,
    S_remaining6_minutes=remain6,O_later_stage_reference_minutes=reference,
    S_remaining_reference_including_collection_minutes=remain6+reference+collect,
    planning_allowance_minutes=10,allowance_not_measured=True,next_check_minutes=110,
    S_later_stages_not_measured=True,final_CPU_and_scientific_acceptance_pending=True)
(snap/'O_TO_S_LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
c=read(cap/'EVIDENCE.json')
assert c['status']=='PASS_NOT_FINAL' and all(c[k]==0 for k in ['payload_reads','forwards','updates','environments','server_writes'])
for name,x in c['volumes'].items():
    assert x['headroom']==x['free']-x['remaining_bound']-x['reserve']>=0
    assert x['reserve']==(2 if name=='/data' else 1)*1024**3
(cap/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(cap/'EVIDENCE.json'),observation=c['observed'],volumes=c['volumes'],no_payload_or_remote_mutation=True),indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
