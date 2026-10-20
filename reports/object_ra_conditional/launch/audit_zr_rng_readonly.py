"""CPU-only comparison of existing RNG snapshots; no model or environment."""
import json, hashlib, pickle, datetime
from pathlib import Path
import torch
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def digest(v):
    if torch.is_tensor(v):
        a = v.detach().cpu().contiguous()
        raw = str((str(a.dtype), tuple(a.shape))).encode() + a.numpy().tobytes()
    elif isinstance(v, np.ndarray):
        raw = str((str(v.dtype), v.shape)).encode() + v.tobytes()
    elif isinstance(v, dict):
        raw = json.dumps([(str(k), digest(v[k])) for k in sorted(v)], separators=(',', ':')).encode()
    elif isinstance(v, (tuple, list)):
        raw = json.dumps([digest(x) for x in v], separators=(',', ':')).encode()
    else:
        raw = pickle.dumps(v, protocol=4)
    return hashlib.sha256(raw).hexdigest()

rows = []
inputs = {}
for seed in (41, 42, 43):
    ra = Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019/ra_coreset/evaluation' % seed)
    zr = Path('/data/mlr_ifa/experiments/Conditional_ZRv2_OBJECT_RA_seed%d_20261019/evaluation' % seed)
    for stage in range(6, 10):
        for task in range(6, stage + 1):
            name = 'rng_before_stage%d_task%d.pth' % (stage, task)
            states = []
            for root in (ra, zr):
                p = root / name
                inputs[str(p)] = sha(p)
                # Trusted, experiment-generated RNG state, explicitly loaded on CPU.
                states.append(torch.load(str(p), map_location='cpu'))
            assert states[0].keys() == states[1].keys()
            components = {str(k): {'ra': digest(states[0][k]), 'zr': digest(states[1][k]),
                                  'equal': digest(states[0][k]) == digest(states[1][k])}
                          for k in states[0]}
            rows.append(dict(seed=seed, stage=stage, task=task, components=components,
                             all_equal=all(x['equal'] for x in components.values())))
assert all(sha(Path(p)) == h for p, h in inputs.items())
assert not torch.cuda.is_initialized()
out = dict(status='PASS_READONLY', time=datetime.datetime.now().astimezone().isoformat(),
           snapshots=len(inputs), cells=len(rows), equal_cells=sum(r['all_equal'] for r in rows),
           cuda_initialized=False, input_hashes=inputs, rows=rows,
           limitation='Cell-start states only; equal initial states do not isolate subsequent per-trajectory noise.')
print(json.dumps(out, indent=2))
