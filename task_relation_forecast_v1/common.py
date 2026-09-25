import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/task_relation_forecast_v1'
STORE = ROOT / 'diagnostic_storage/task_relation_forecast_v1'
TASKS = (6, 7, 8, 9)
SEEDS = (45, 46, 47)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stable_seed(*items):
    return int(hashlib.sha256('/'.join(map(str, items)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, ('Frozen artifact changed', str(path))
    else:
        write(path, value)


def read(path):
    return json.loads(Path(path).read_text())


def split_demos(task, demos):
    ordered = sorted(demos, key=lambda d: (hashlib.sha256(('task_relation_forecast_v1/split/%s/%s' % (task, d)).encode()).hexdigest(), d))
    n = len(ordered)
    assert 5 <= n <= 50, ('Unexpected demonstration count', n)
    a, d = (30, 10) if n == 50 else (int(.6 * n), int(.2 * n))
    return dict(A_train=ordered[:a], D_signal=ordered[a:a+d], E_eval=ordered[a+d:])


def balanced_windows(demos, lengths, cap):
    import numpy as np
    quotas = {d: min(lengths[d], cap // len(demos)) for d in demos}
    for d in demos:
        if sum(quotas.values()) >= cap:
            break
        quotas[d] = min(lengths[d], quotas[d] + 1)
    return [(d, int(i)) for d in demos for i in np.linspace(0, lengths[d]-1, quotas[d], dtype=int)]
