"""Validate unchanged configuration and task6 negative control, without edits."""
import csv
import json
import math
import re
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
seed = int(sys.argv[1])
old = root / f'runs/final_paper_text_faithful/ifa/seed{seed}/run_001'
new = root / f'runs/own_reference_control/ifa/seed{seed}/run_001'
a = json.loads((old / 'config.json').read_text())
b = json.loads((new / 'config.json').read_text())
for key in ('seed', 'train', 'gamma', 'ifa_weight', 'past_data_rate', 'feature_dir', 'pretrain', 'repro_scheduler_mode', 'start_task', 'end_task'):
    assert a[key] == b[key], (key, a[key], b[key])
assert b['ifa_own_reference_fix'] is True
assert len(list(new.glob('mlrifa_*_model_ep*.pth'))) == 28
rows = list(csv.DictReader((new / 'paper_repro_train_curve.csv').open()))
assert len(rows) == 400, len(rows)
assert all(math.isfinite(float(r['mean_bc_loss'])) for r in rows)
old_buffer = Path(a['buffer_dir'])
new_buffer = Path(b['buffer_dir'])
assert {p.name for p in old_buffer.glob('*.pt')} == {p.name for p in new_buffer.glob('*.pt')}
pair_pattern = re.compile(r'\[protocol\].*stage=(\d+) ifa_enabled=True selected_pairs=(\[.*\])')
old_pairs = pair_pattern.findall((root / f'runs/final_paper_text_faithful_driver/train_ifa_seed{seed}.log').read_text())
new_pairs = pair_pattern.findall((root / f'runs/own_reference_control_driver/train_seed{seed}.log').read_text())
assert len(old_pairs) == 4 and old_pairs == new_pairs, (old_pairs, new_pairs)
before = torch.load(old / 'mlrifa_6_model_ep99.pth', map_location='cpu')['state_dict']
after = torch.load(new / 'mlrifa_6_model_ep99.pth', map_location='cpu')['state_dict']
assert before.keys() == after.keys()
changed = [k for k in before if not torch.equal(before[k], after[k])]
print(json.dumps({'seed': seed, 'config_match': True, 'finite_400_epochs': True, 'buffer_selection_match': True, 'selected_pairs_match': old_pairs, 'task6_changed_tensors': len(changed), 'task6_max_abs_difference': max((float((before[k]-after[k]).abs().max()) for k in changed), default=0)}, indent=2))
assert not changed, 'Task6 negative control differs: investigate before evaluation'
