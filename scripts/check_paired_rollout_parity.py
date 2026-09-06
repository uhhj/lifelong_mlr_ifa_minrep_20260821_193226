"""Ensure full observation does not change any rollout action or done flag."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/paired_rollout_diagnosis'
for round_id in range(2):
    with np.load(RUN / f'parity_light/round{round_id}.npz') as light, np.load(RUN / f'seed43_stage7_task7/mlr/round{round_id}.npz') as full:
        for key in ('initial_states', 'actions', 'done', 'steps'):
            assert np.array_equal(light[key], full[key]), (round_id, key)
out = ROOT / 'reports/paired_rollout_diagnosis/INSTRUMENTATION_PARITY.json'
out.write_text(json.dumps(dict(verdict='PASS', trials=20,
    identical=['initial_states', 'all actions', 'all done flags', 'all step indices'],
    scope='Light wrapper only copies returned actions/dones. Full wrapper additionally queries sim state, copies numeric observations and captures sparse images. Both delegate unchanged official evaluator and vector environment.'), indent=2)+'\n')
print('INSTRUMENTATION_PARITY_PASS', flush=True)
