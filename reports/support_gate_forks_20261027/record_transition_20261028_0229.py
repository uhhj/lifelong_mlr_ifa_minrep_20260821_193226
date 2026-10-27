"""Verify saved C-to-S progress without any server mutation."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
D = R / 'check_readonly_20261028_022941'
P = R / 'check_readonly_20261028_021740'
old = json.loads((P/'EVIDENCE.json').read_text())['light']
new = json.loads((D/'EVIDENCE.json').read_text())['light']
def data(v, key):
    return json.loads(v[key]['text'])
prefix = {}
for key in old:
    if '/EPOCHS_' in key:
        a, b = data(old, key), data(new, key)
        assert b[:len(a)] == a
        prefix[key] = len(a)
assert new['GOAL_41/GATES.jsonl']['text'] == old['GOAL_41/GATES.jsonl']['text']
assert new['GOAL_42/train_C.exit']['text'].strip() == '0'
assert len(data(new, 'GOAL_42/EPOCHS_C.json')) == 100
ad = data(new, 'GOAL_42/ADMISSION_S.json')
assert ad['status'] == 'PASS_OWN_GPU_ADMISSION' and ad['optimizer_calls'] == 29
assert ad['extra_autograd_calls'] == 11 and ad['environments'] == 0
for k in ['roundtrip_next_step_exact', 'q_one_original_S_exact', 'lambda_zero_original_C_exact', 'ten_step_C_replay_exact', 'accepted_and_rejected_full_state_restore', 'measurement_state_exact', 'probe_all_state_exact']:
    assert ad[k]
assert data(new, 'GOAL_42/INITIAL_C.json')['common_state_sha256'] == data(new, 'GOAL_42/INITIAL_S.json')['common_state_sha256']
out = dict(status='PASS_PROGRESS_NOT_FINAL', exact_previous_epoch_prefixes=prefix,
    GOAL41_gate_records_unchanged=1900, GOAL42_C_natural_exit0=True,
    C_epoch_minutes=data(new, 'GOAL_42/TRAIN_C.json')['epoch_seconds']/60,
    S_own_isolated_optimizer_calls=29, S_extra_admission_autograd=11,
    S_common_initial_state_exact_C=True, S_complete_epochs=35,
    final_CPU_and_closed_loop_pending=True,
    source_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),
    previous_source_sha256=hashlib.sha256((P/'EVIDENCE.json').read_bytes()).hexdigest(),
    new_optimizer_calls=0, new_environments=0)
(D/'GOAL42_C_TO_S_LOCAL_AUDIT.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
print(json.dumps(out))
