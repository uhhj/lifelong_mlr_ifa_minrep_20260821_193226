"""Additional isolated checks using existing real smoke features; no formal data."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from libero.lifelong.runbook_core import transactional_buffer_update, atomic_json
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed, make_linear_schedule_with_warmup


def digest(model, prefix):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        if key.startswith(prefix):
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-name', default='additional')
    args = p.parse_args()
    smoke = args.run_dir / 'smoke'
    out = smoke / args.output_name
    out.mkdir(exist_ok=False)
    empty = out / 'empty_buffer'
    empty.mkdir()
    original = sorted(p.name for p in (smoke / 'features/6').glob('*.pt'))
    result = transactional_buffer_update(smoke / 'features', 6, empty, out / 'generations')
    again = transactional_buffer_update(smoke / 'features', 6, empty, out / 'generations')
    assert result == again and len(result['files']) == 1
    assert not list(empty.iterdir())
    assert original == sorted(p.name for p in (smoke / 'features/6').glob('*.pt'))
    # Original seed42 chooses one of the two windows; verify exact selection.
    import random
    expected = random.Random(42).sample(original, 1)[0]
    assert result['files'][0]['name'] == 't6__' + expected
    control_seed(42)
    saved = torch.load(str(smoke / 'epoch_boundary_smoke.pth'), map_location='cpu')
    cfg = saved['cfg']
    model = BriefPolicy(cfg, cfg.shape_meta).cuda().train()
    params = [n for n, v in model.named_parameters() if v.requires_grad]
    assert params and all(n.startswith(('temporal_transformer.', 'policy_head.')) for n in params)
    before = {k: digest(model, k) for k in ['temporal_transformer.', 'policy_head.']}
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.train.optimizer.kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_linear_schedule_with_warmup(500, 2000))
    sample = torch.load(str(smoke / 'features/6/0.pt'), map_location='cuda')
    lrs = []
    for _ in range(2):
        optimizer.zero_grad()
        distribution, _ = model(sample['feature'])
        loss = model.policy_head.loss_fn(distribution, sample['actions'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    changed = {k: digest(model, k) != before[k] for k in before}
    assert all(changed.values()) and lrs[0] == 0 and lrs[1] > 0
    from torch.utils.data import DataLoader, DistributedSampler
    from libero.lifelong.algos.mlr import FileTrajectoryDataset
    dataset = FileTrajectoryDataset(sorted(str(p) for p in (smoke/'features/6').glob('*.pt')))
    signatures = []
    for workers in (0, 12):
        control_seed(42)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        sampler.set_epoch(3)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=workers)
        values = []
        for sample, tids in loader:
            values.append(hashlib.sha256(sample['feature'].numpy().tobytes()).hexdigest())
        signatures.append((values, torch.get_rng_state()))
    assert signatures[0][0] == signatures[1][0]
    assert torch.equal(signatures[0][1], signatures[1][1])
    atomic_json(out / 'result.json', {'status': 'PASS', 'kind': 'SOFTWARE_SMOKE_ONLY',
        'workers_0_vs_12_same_order_and_parent_torch_rng': True, 'buffer_same_call_idempotent': True, 'original_buffer_untouched': True,
        'selection_matches_original_seed42': True, 'buffer_selection': result,
        'updated_modules': changed, 'learner_trainable_names': params, 'lr_used': lrs})
    print('ADDITIONAL_CHECKS_PASS')


if __name__ == '__main__':
    main()
