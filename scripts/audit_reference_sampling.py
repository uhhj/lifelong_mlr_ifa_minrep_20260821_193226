"""Read-only reconstruction: filenames, sampler indices, and existing run logs."""
import ast
import csv
import json
import math
import random
import re
from pathlib import Path

import torch
from torch.utils.data.distributed import DistributedSampler

root = Path('/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep')
results = []
for seed in (42, 43, 44):
    run = root / f'runs/final_paper_text_faithful/ifa/seed{seed}/run_001'
    cfg = json.loads((run / 'config.json').read_text())
    assert cfg['seed'] == seed and cfg['train']['batch_size'] == 10
    assert cfg['train']['n_epochs'] == 100 and cfg['past_data_rate'] == .5
    assert cfg['repro_scheduler_mode'] == 'batch_step' and cfg['gamma'] == .30
    feature = Path(cfg['feature_dir'])
    buffer = Path(cfg['buffer_dir'])
    log = (root / f'runs/final_paper_text_faithful_driver/train_ifa_seed{seed}.log').read_text()
    pairs = {int(stage): ast.literal_eval(value) for stage, value in re.findall(r'stage=(\d+) ifa_enabled=True selected_pairs=(\[[^\n]*\])', log)}
    lr = list(csv.DictReader((run / 'lr_protocol.csv').open()))
    curve = list(csv.DictReader((run / 'paper_repro_train_curve.csv').open()))
    assert len(lr) == len(curve) == 400
    assert all(math.isfinite(float(row['mean_bc_loss'])) for row in curve)
    virtual = []
    for stage in range(6, 10):
        current = sorted(str(p) for p in (feature / str(stage)).glob('*.pt'))
        assert current
        mixed = sorted(virtual) + current
        random.Random(seed).shuffle(mixed)
        ids = [int(re.match(r't(\d+)__', Path(p).name)[1]) if Path(p).name.startswith('t') else int(Path(p).parent.name) for p in mixed]
        sampler = DistributedSampler(ids, num_replicas=1, rank=0, shuffle=True)
        assert sampler.seed == 0 and sampler.drop_last is False
        old = {j if k == stage else k for j, k in pairs[stage]}
        totals = dict(all_batches=0, eligible_batches=0, wrong_reference_batches=0, eligible_current_samples=0, affected_current_samples=0, first_equals_negative_task=0)
        for epoch in range(100):
            rows = [r for r in lr if int(r['stage']) == stage and int(r['epoch']) == epoch]
            assert len(rows) == 1
            row = rows[0]
            count = math.ceil(len(ids) / 10)
            assert int(row['batches']) == count
            assert int(row['optimizer_step_start']) == epoch * count
            assert int(row['optimizer_step_end']) == (epoch + 1) * count
            assert int(row['total_steps']) == count * 100
            assert int(row['batch_size']) == 10 and int(row['warmup_steps']) == 500
            sampler.set_epoch(epoch)
            order = list(sampler)
            for offset in range(0, len(order), 10):
                batch = [ids[i] for i in order[offset:offset + 10]]
                totals['all_batches'] += 1
                ncur = batch.count(stage)
                if old and ncur:
                    totals['eligible_batches'] += 1
                    totals['eligible_current_samples'] += ncur
                    if batch[0] != stage:
                        totals['wrong_reference_batches'] += 1
                        totals['affected_current_samples'] += ncur
                    if batch[0] in old:
                        totals['first_equals_negative_task'] += 1
        selected = random.Random(seed).sample(current, max(1, math.ceil(len(current)*.5)))
        assert len(virtual) + len(selected) <= 59660000
        virtual.extend(str(buffer / f't{stage}__{Path(p).name}') for p in selected)
        result = dict(seed=seed, stage=stage, selected_pairs=pairs[stage], current_files=len(current), dataset_files=len(ids), **totals)
        result['wrong_reference_percent'] = round(100*totals['wrong_reference_batches']/totals['eligible_batches'], 4) if totals['eligible_batches'] else None
        result['affected_current_sample_percent'] = round(100*totals['affected_current_samples']/totals['eligible_current_samples'], 4) if totals['eligible_current_samples'] else None
        results.append(result)
    actual = sorted(str(p) for p in buffer.glob('*.pt'))
    assert sorted(virtual) == actual, f'Final replay filename mismatch seed{seed}'
print(json.dumps(dict(torch=torch.__version__, sampler_seed=0, sampler_drop_last=False, reconstruction='deterministic index reconstruction; original per-batch IDs were not logged', validation='all 1200 stage-epoch batch/update counts match; losses finite; reconstructed final replay filenames exactly match all three formal buffers', stages=results), indent=2))
