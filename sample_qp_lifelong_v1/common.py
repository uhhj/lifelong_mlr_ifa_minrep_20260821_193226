import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/sample_qp_lifelong_v1'
RUN = ROOT/'runs/sample_qp_lifelong_v1/seed46/sample_qp'
SEED = 46
TASKS = (6, 7, 8, 9)
STORAGE = Path('/root/sample_qp_lifelong_v1')
SAVED_EPOCHS = (70, 75, 80, 85, 90, 95, 99)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def publish_eval_log(source, destination):
    from fixed_kappa_experiment.metrics import read_fixed_matrix
    matrix=read_fixed_matrix(source)
    destination=Path(destination)
    if destination.exists():
        assert read_fixed_matrix(destination)==matrix, 'Do not replace a completed evaluation'
    else:
        tmp=destination.with_name(destination.name+'.tmp')
        tmp.write_bytes(Path(source).read_bytes());os.replace(tmp,destination)
    return matrix

def verify():
    p = read(OUT/'PROTOCOL_FROZEN.json')
    for f,h in p['files'].items():
        assert sha(ROOT/f) == h, ('Frozen input changed', f)
    return p

def selected_references(plan):
    """Deterministic uniform-without-replacement window choice; no global RNG."""
    import random
    selected = {}
    for task in range(6, plan['task']):
        eligible = sorted([x for x in plan['records'] if x['task'] == task],
                          key=lambda x:x['origin'])
        assert len(eligible) >= 32, ('Insufficient retained windows', task)
        key = 'sample_qp_lifelong_v1/46/%d/%d' % (plan['task'], task)
        private_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        selected[str(task)] = random.Random(private_seed).sample(eligible, 32)
    return selected

class Guard:
    def __init__(self):
        self.last = 0
    def check(self, force=False):
        import torch, shutil
        from functional_geometry_diagnostic_v1.runtime_guard import memory
        if not force and time.monotonic()-self.last < 30: return
        self.last = time.monotonic(); m = memory()
        assert m['rss'] < 28*2**30 and m['available'] > 2*2**30, 'Host memory budget'
        assert torch.cuda.memory_allocated() < 20*2**30, 'GPU allocated budget'
        assert torch.cuda.memory_reserved() < 22*2**30, 'GPU reserved budget'
        assert shutil.disk_usage(ROOT).free > 1*2**30, 'Data disk reserve'
        assert shutil.disk_usage(STORAGE).free > 4*2**30, 'Root disk reserve'

def config(destination):
    from cb_ifa_experiment.gpu_runtime import train_config
    cfg = train_config(SEED, destination)
    cfg.train.n_epochs = 100
    cfg.start_task = 6; cfg.end_task = 10
    cfg.save_final_checkpoint = True
    cfg.repro_arm = 'sample_qp_lifelong_v1'
    assert cfg.train.batch_size == 10 and cfg.train.num_workers == 0
    assert cfg.train.grad_clip == 100 and cfg.train.loss_scale == 1
    assert cfg.repro_scheduler_mode == 'batch_step' and cfg.past_data_rate == .5
    assert not cfg.ifa_enabled and not cfg.aug
    return cfg

def dataset_loader(task):
    from cb_ifa_experiment.gpu_runtime import CachedWindows, torch
    plan = read(OUT/('plans/task%d.json' % task))
    ds = CachedWindows([x['origin'] for x in plan['records']],
                       [x['sha256'] for x in plan['records']])
    sampler = torch.utils.data.DistributedSampler(ds, num_replicas=1, rank=0,
                                                 shuffle=True, seed=0, drop_last=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=10, sampler=sampler,
                                        num_workers=0, drop_last=False)
    assert len(loader) == plan['batches_per_epoch']
    return plan, ds, sampler, loader

def reference_batches(algo, task):
    from cb_ifa_experiment.gpu_runtime import torch
    from torch.utils.data._utils.collate import default_collate
    refs = read(OUT/('plans/references_task%d.json' % task))
    batches = []
    for old in sorted(refs, key=int):
        rows = refs[old]
        assert len(rows) == 32 and int(old) < task
        for offset in range(0, 32, 8):
            samples = []
            for row in rows[offset:offset+8]:
                assert sha(ROOT/row['origin']) == row['sha256']
                samples.append(torch.load(ROOT/row['origin'], map_location='cpu'))
            batches.append(algo.map_tensor_to_device(default_collate(samples)))
    return batches
