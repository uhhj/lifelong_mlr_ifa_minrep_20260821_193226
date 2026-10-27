"""Compare immutable prefixes and own admissions in two read-only snapshots."""
import hashlib
import json
import sys
from pathlib import Path

previous, current = map(Path, sys.argv[1:3])
old = json.loads((previous/'EVIDENCE.json').read_text())['light']
new = json.loads((current/'EVIDENCE.json').read_text())['light']
def read(v, key):
    return json.loads(v[key]['text'])
prefixes = {}
for key in old:
    if '/EPOCHS_' in key:
        a, b = read(old, key), read(new, key)
        assert b[:len(a)] == a, key
        prefixes[key] = len(a)
    elif key.endswith('/GATES.jsonl'):
        a, b = old[key]['text'].splitlines(), new[key]['text'].splitlines()
        assert b[:len(a)] == a, key
new_admissions = {}
for key in new:
    if '/ADMISSION_' not in key or key in old:
        continue
    ad = read(new, key)
    assert ad['status'] == 'PASS_OWN_GPU_ADMISSION'
    assert ad['optimizer_calls'] == 29 and ad['extra_autograd_calls'] == 11
    assert ad['environments'] == 0
    for field in ['roundtrip_next_step_exact', 'q_one_original_S_exact', 'lambda_zero_original_C_exact', 'ten_step_C_replay_exact', 'accepted_and_rejected_full_state_restore', 'measurement_state_exact', 'probe_all_state_exact']:
        assert ad[field], (key, field)
    initial = key.replace('ADMISSION_', 'INITIAL_')
    reference = key.split('/')[0]+'/INITIAL_C.json'
    assert read(new, initial)['common_state_sha256'] == read(new, reference)['common_state_sha256']
    new_admissions[key] = dict(optimizer_calls=29, autograd_calls=11, common_state_equal_C=True)
completed = {}
for key in new:
    if '/train_' in key and key.endswith('.exit'):
        assert new[key]['text'].strip() == '0', key
        meta = read(new, key.replace('train_', 'TRAIN_').replace('.exit', '.json'))
        assert meta['epochs'] == 100
        completed[key] = dict(epochs=100, epoch_minutes=meta['epoch_seconds']/60)
out = dict(status='PASS_PROGRESS_NOT_FINAL', previous_epoch_prefixes_exact=prefixes,
           new_own_admissions=new_admissions, completed_exit0=completed,
           source_sha256=hashlib.sha256((current/'EVIDENCE.json').read_bytes()).hexdigest(),
           previous_source_sha256=hashlib.sha256((previous/'EVIDENCE.json').read_bytes()).hexdigest(),
           final_CPU_and_closed_loop_pending=True, new_optimizer_calls=0, new_environments=0)
(current/'TRANSITION_AUDIT.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
print(json.dumps(out))
