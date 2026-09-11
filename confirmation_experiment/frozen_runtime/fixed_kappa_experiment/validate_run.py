"""Require the complete prescribed run, never require performance or projection hits."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

from metrics import diagnostic_summary
from protocol import check_fixed_recipe, common_recipe, read, require, sha, source_hashes


def validate(run, kappa=.5, smoke=False):
    manifest = read(run / 'experiment.json')
    require(manifest['kappa'] == kappa and manifest['smoke'] == smoke, 'Wrong coefficient/phase')
    require((run / 'TRAINING_COMPLETE').is_file(), 'Training incomplete')
    cfg = read(run / 'run_001/config.json')
    repo = run.resolve().parents[5]
    require(manifest['source_sha256'] == source_hashes(repo), 'Source changed since training')
    require(manifest['references']['initial_checkpoint_sha256'] == sha(Path(manifest['checkpoint'])),
            'Initial checkpoint changed since training')
    check_fixed_recipe(cfg, manifest['seed'], smoke)
    require(cfg['lifelong']['algo'] == 'MLRFixedKappa', 'Wrong algorithm')
    require(cfg['lifelong']['projection_enabled'] is True, 'Controller disabled')
    require(cfg['lifelong']['kappa'] == kappa, 'Wrong saved coefficient')
    require(cfg['lifelong']['reference_norm_floor'] == 1e-12, 'Reference threshold changed')
    recipe = copy.deepcopy(common_recipe(cfg))
    if smoke:
        recipe['train']['n_epochs'] = 100
        recipe['end_task'] = 10
    require(recipe == manifest['references']['common_recipe'], 'Common recipe differs from endpoints')
    path = run / 'run_001/fixed_kappa_epochs.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {(s, e) for s in range(6, 8 if smoke else 10) for e in range(1 if smoke else 100)}
    keys = [(int(r['stage']), int(r['epoch'])) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Missing/duplicate epoch records')
    for r in rows:
        require(float(r['kappa']) == kappa and r['enabled'] == 'True', 'Coefficient changed during training')
        require(int(r['batches']) > 0, 'Empty epoch')
        for value in r.values():
            if value and value not in ('True', 'False'):
                require(math.isfinite(float(value)), 'Nonfinite epoch metric')
        if kappa == 0:
            require(int(r['projected_batches']) == 0 and float(r['mean_correction_ratio']) == 0,
                    'Zero coefficient changed a gradient')
    for stage in range(7, 8 if smoke else 10):
        require(sum(int(r['reference_batches']) for r in rows if int(r['stage']) == stage) > 0,
                'No replay reference in stage %d' % stage)
    ep = 0 if smoke else 99
    for stage in range(6, 8 if smoke else 10):
        ckpt = run / 'run_001' / ('mlrifa_%d_model_ep%d.pth' % (stage, ep))
        require(ckpt.is_file() and ckpt.stat().st_size > 0, 'Missing checkpoint')
    result = {'status': 'PASS', 'kappa': kappa, 'smoke': smoke, 'performance_gate': False,
        'diagnostics': diagnostic_summary(path, expected_grid=expected)}
    (run / 'protocol_validation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--kappa', type=float, default=.5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    print(json.dumps(validate(a.run, a.kappa, a.smoke), indent=2))
