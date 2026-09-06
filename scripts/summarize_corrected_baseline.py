"""Descriptive aggregates of the fixed probe grid; no statistical tests."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/corrected_baseline_diagnosis'


def read(name):
    return list(csv.DictReader((OUT / name).open()))


def aggregate(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    results = []
    for key, members in sorted(grouped.items()):
        result = dict(zip(keys, key), n_probe_batches=len(members))
        for metric in metrics:
            values = [float(m[metric]) for m in members if m[metric] != '']
            result[metric + '_mean'] = statistics.mean(values) if values else None
            if metric == 'cosine':
                result['nonzero_gradient_batches'] = len(values)
                result['negative_cosine_fraction'] = sum(v < 0 for v in values)/len(values) if values else None
        results.append(result)
    return results


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    complete = json.loads((OUT / 'COMPLETE.json').read_text())
    assert complete['status'] == 'DIAGNOSTIC_COMPLETE' and complete['optimizer_steps'] == 0
    bc, pairs, ifa = (read(n) for n in ('bc_profiles.csv', 'bc_gradient_pairs.csv', 'ifa_gradient_pairs.csv'))
    assert (len(bc), len(pairs), len(ifa)) == (1440, 2160, 1920)
    write('bc_by_bin.csv', aggregate(bc, ('arm', 'seed', 'checkpoint_stage', 'task', 'bin'), ('bc',)))
    by_task = aggregate(bc, ('arm', 'seed', 'checkpoint_stage', 'task'), ('bc',))
    write('bc_by_task.csv', by_task)
    write('bc_pair_by_bin.csv', aggregate(pairs, ('arm', 'seed', 'checkpoint_stage', 'task_a', 'task_b', 'bin'), ('cosine',)))
    write('ifa_pair_by_bin.csv', aggregate(ifa, ('arm', 'seed', 'checkpoint_stage', 'current', 'selected_old', 'target', 'bin'), ('cosine', 'weighted_ifa', 'hinge_active')))
    ifa_stage = aggregate([r for r in ifa if r['current'] == r['checkpoint_stage']],
                          ('arm', 'seed', 'current', 'selected_old', 'target'), ('cosine', 'weighted_ifa', 'hinge_active'))
    write('ifa_at_pair_stage.csv', ifa_stage)
    index = {(r['arm'], r['seed'], r['checkpoint_stage'], r['task']): r['bc_mean'] for r in by_task}
    differences = []
    for seed in ('42', '43', '44'):
        for stage in ('6', '7', '8', '9'):
            for task in ('7', '9'):
                for control in ('mlr', 'release_ifa'):
                    differences.append(dict(seed=seed, stage=stage, task=task, control=control,
                        corrected_bc=index['corrected_ifa', seed, stage, task],
                        control_bc=index[control, seed, stage, task],
                        corrected_minus_control=index['corrected_ifa', seed, stage, task]-index[control, seed, stage, task]))
    write('task7_task9_bc_deltas.csv', differences)
    (OUT / 'SUMMARY.json').write_text(json.dumps(dict(task7_task9_deltas=differences,
        ifa_at_pair_stage=ifa_stage, interpretation='BC negative log likelihood and gradient cosine on in-distribution cached demonstration probes; neither rollout success nor actual AdamW step. Negative cosine predicts opposition only under an infinitesimal plain gradient step. Batches are not independent statistical replicates.'), indent=2)+'\n')
    print(json.dumps(dict(task7_task9_deltas=differences, corrected_ifa_at_pair_stage=[r for r in ifa_stage if r['arm']=='corrected_ifa']), indent=2))


if __name__ == '__main__':
    main()
