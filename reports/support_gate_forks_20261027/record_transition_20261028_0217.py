"""Read-only local prefix and transition verification for the saved snapshot."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
D = R / 'check_readonly_20261028_021740'
P = R / 'check_readonly_20261028_010017'
old = json.loads((P / 'EVIDENCE.json').read_text())['light']
new = json.loads((D / 'EVIDENCE.json').read_text())['light']
def data(v, key):
    return json.loads(v[key]['text'])
prefix = {}
for arm in 'CSQG':
    key = 'GOAL_41/EPOCHS_' + arm + '.json'
    a, b = data(old, key), data(new, key)
    assert b[:len(a)] == a
    assert len(b) == 100
    assert new['GOAL_41/train_' + arm + '.exit']['text'].strip() == '0'
    prefix[arm] = len(a)
old_gates = [json.loads(x) for x in old['GOAL_41/GATES.jsonl']['text'].splitlines()]
new_gates = [json.loads(x) for x in new['GOAL_41/GATES.jsonl']['text'].splitlines()]
assert new_gates[:len(old_gates)] == old_gates
assert len(new_gates) == 1900
admission = data(new, 'GOAL_42/ADMISSION_C.json')
assert admission['status'] == 'PASS_OWN_GPU_ADMISSION'
assert admission['optimizer_calls'] == 29
assert all(admission[k] for k in ['roundtrip_next_step_exact', 'q_one_original_S_exact', 'lambda_zero_original_C_exact', 'ten_step_C_replay_exact', 'accepted_and_rejected_full_state_restore', 'measurement_state_exact', 'probe_all_state_exact'])
train = data(new, 'GOAL_41/TRAIN_G.json')
assert train['gate_blocks'] == 1900 and train['Q_accepts'] == 1085
out = dict(status='PASS_PROGRESS_NOT_FINAL', previous_epoch_prefixes_exact=prefix,
           previous_gate_records_exact=len(old_gates), GOAL41_all_four_natural_exit0=True,
           G_epoch_minutes=train['epoch_seconds']/60, G_blocks=1900, G_Q_accepts=1085,
           G_accept_fraction=1085/1900, GOAL42_C_own_optimizer_admission=29,
           GOAL42_C_complete_epochs=81, final_CPU_and_closed_loop_pending=True,
           source_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),
           previous_source_sha256=hashlib.sha256((P/'EVIDENCE.json').read_bytes()).hexdigest(),
           new_optimizer_calls=0, new_environments=0)
(D/'GOAL41_TO_GOAL42_LOCAL_AUDIT.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
print(json.dumps(out))
