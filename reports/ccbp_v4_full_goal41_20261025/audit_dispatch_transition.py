"""Read-only local audit of inherited D completion and actual short-history admission."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

snap, cap = map(Path, sys.argv[1:3])
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
v = read(snap/'EVIDENCE.json')
light = v['light']
load = lambda n: json.loads(light[n]['text'])
for n, x in light.items():
    assert hashlib.sha256(x['text'].encode()).hexdigest() == x['sha256'], n
inherit = load('INHERITED_D_EXIT_OBSERVATION.json')
assert inherit['status'] == 'PASS_OBSERVED_NATURAL_D_EXIT'
assert inherit['wait_status']['state'] == 'Z' and inherit['wait_status']['exit_code'] == 0
assert inherit['wait_status']['starttime'] == '2007758'
assert inherit['epochs'] == 400 and inherit['steps'] == 76100
assert inherit['completed_epochs_repeated'] == inherit['collection_attempts_repeated'] == 0
assert inherit['source_contract'] == '8edb70c911328d8440f25dea66f719c8259a4c332bef5be5a55caec84aea452c'
for n, h in inherit['files'].items():
    assert light[n]['sha256'] == h, n
assert load('DISPATCHER_RETIRED.json')['old_controller_not_claimed_exit0']
assert not {18453, 18454, 18459} & {p['pid'] for p in v['processes']}
assert light['train_D.exit']['text'].strip() == '0'
train = load('TRAIN_D.json')
hist = load('EPOCHS_D.json')
assert len(hist) == 400 and abs(sum(x['seconds'] for x in hist)-train['summed_epoch_seconds']) < 1e-8
for t in range(6,10):
    assert light['collect_D_%d.exit'%t]['text'].strip() == '0'
    assert load('COLLECTION_STATE_PARITY_D_%d.json'%t)['parent_all_state_exact']
    assert train['checkpoints'][str(t)] == load('STAGE_D_%d.json'%t)['sha256']
short = load('SHORT_HISTORY_ADMISSION_O.json')
assert short['status'] == 'PASS' and short['eight_frame_legacy_exact'] and short['all_state_restored']
assert [r['length'] for r in short['lengths']] == list(range(1,9))
assert all(r['finite_gradient'] and r['actual_slots']==4 for r in short['lengths'])
assert short['extra_autograd_calls']==8 and short['policy_calls']==17
assert short['optimizer_calls']==short['environments']==0
events = [json.loads(s) for s in light['COLLECTION_EVENTS.jsonl']['text'].splitlines() if s]
dcomplete = [e for e in events if e['kind']=='complete' and e['arm']=='D']
assert len(dcomplete)==4 and sum(e['attempts'] for e in dcomplete)==40
audit = read(snap/'LOCAL_AUDIT.json')
remaining6 = (100-audit['counts']['O']['by_stage']['6'])*audit['rates_last10']['O']['6']/60
later_reference = sum(x['seconds'] for x in hist if x['stage']>6)/60
collection_reference = sum(x['seconds'] for x in dcomplete)/60
out = dict(status='PASS_PROGRESS_NOT_FINAL', evidence_sha256=sha(snap/'EVIDENCE.json'),
           observation=v['observed'], inherited_files_verified=len(inherit['files']),
           inherited_D_natural_exit=True, old_dispatcher_not_claimed_exit0=True,
           completed_epochs_repeated=0, collection_attempts_repeated=0,
           O_short_history_actual_PASS=True, S_short_history='PENDING',
           D_epoch_minutes=train['summed_epoch_seconds']/60,
           D_collection_seconds=sum(x['seconds'] for x in dcomplete),
           O_remaining6_minutes=remaining6, D_later_stages_reference_minutes=later_reference,
           O_remaining_reference_including_collection_minutes=remaining6+later_reference+collection_reference,
           planning_allowance_minutes=10, allowance_not_measured=True, next_check_minutes=105,
           no_partial_scientific_performance=True)
(snap/'TRANSITION_LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
c = read(cap/'EVIDENCE.json')
assert c['status']=='PASS_NOT_FINAL'
assert all(c[k]==0 for k in ['payload_reads','forwards','updates','environments','server_writes'])
for name, x in c['volumes'].items():
    assert x['headroom']==x['free']-x['remaining_bound']-x['reserve']>=0
    assert x['reserve']==(2 if name=='/data' else 1)*1024**3
(cap/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(cap/'EVIDENCE.json'),observation=c['observed'],volumes=c['volumes'],no_payload_or_remote_mutation=True),indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
