"""Read-only retrospective: stream saved events; never build a model or train.

Native sampling is reconstructed, not represented as historical per-step logs.
Its permutation/current-slot hashes are checked against all saved epoch records.
Existing feature tensors are read only to verify the IFA language-anchor identity.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()


def write(p, data):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(str(tmp), str(p))


def csvwrite(p, rows):
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(str(tmp), str(p))


def task_of(identity):
    return int(identity.split('__')[0][1:])


def stats(values):
    return dict(min=float(min(values)), median=float(np.median(values)),
                max=float(max(values)), mean=float(np.mean(values)))


def main(root, out):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'AUDIT.json').exists(), 'Preserve existing audit; use a new output directory.'
    report = root / 'reports/v1_object_tsr'
    run = root / 'runs/v1_object_tsr/seed42'
    features = root / 'features/object_seed42'
    routes = read(report / 'ROUTING_SUMMARY.json')
    provenance, stages, replay_rows, coverage_rows, epoch_rows, anchor_rows = [], {}, [], [], [], []
    language_checks = []
    validation = Counter()
    for task in range(6, 10):
        old_dir = run / ('transactions/task%d/buffer' % (task - 1))
        old_paths = sorted(old_dir.glob('*.pt')) if task > 6 else []
        current_paths = sorted((features / str(task)).glob('*.pt'))
        filepaths = old_paths + current_paths
        random.Random(42).shuffle(filepaths)
        identities = [p.name if p.name.startswith('t') else 't%d__%s' % (task, p.name) for p in filepaths]
        group_ids = defaultdict(list)
        for identity in identities:
            group_ids[task_of(identity)].append(identity)
        total = Counter()
        native_anchors, actual_anchors, pair_counter = Counter(), Counter(), Counter()
        native_ifa_anchors, actual_ifa_anchors = Counter(), Counter()
        anchor_buckets = defaultdict(Counter)
        per_id = Counter()
        per_epoch_coverage = defaultdict(list)
        used_language = defaultdict(set)
        source_to_path = dict(zip(identities, filepaths))
        s = Counter()
        for epoch in range(100):
            g = torch.Generator().manual_seed(epoch)
            order = torch.randperm(len(identities), generator=g).tolist()
            sample = read(report / ('seed42/sampling/task%d_epoch%03d.json' % (task, epoch)))
            assert digest(order) == sample['original_index_sha256'], (task, epoch, 'permutation')
            curr_order = [i for i in order if task_of(identities[i]) == task]
            assert digest(curr_order) == sample['current_index_sha256'], (task, epoch, 'current order')
            validation['epoch_hashes_matched'] += 1
            log = report / ('seed42/events/task%d_epoch%03d.jsonl' % (task, epoch))
            provenance.append(dict(path=str(log.relative_to(root)), size=log.stat().st_size, sha256=sha(log)))
            ec, ic, es = Counter(), Counter(), Counter()
            with log.open() as f:
                for bi, line in enumerate(f):
                    e = json.loads(line)
                    assert (e['task'], e['epoch']) == (task, epoch)
                    assert e['step'] == epoch * ((len(order) + 9) // 10) + bi + 1
                    native = [identities[i] for i in order[bi * 10:(bi + 1) * 10]]
                    actual = e['ids']
                    assert len(native) == len(actual)
                    assert [task_of(x) for x in actual] == e['source_tasks']
                    for n, a in zip(native, actual):
                        assert (task_of(n) == task) == (task_of(a) == task)
                        if task_of(n) == task:
                            assert n == a
                        elif n != a:
                            validation['changed_old_slots'] += 1
                    if task < 8:
                        assert native == actual
                    validation['batches_current_slots_and_ids_matched'] += 1
                    nc = e['current_count']
                    assert nc == sum(task_of(x) == task for x in actual)
                    ec.update(e['source_tasks']); ic.update(actual)
                    nanchor, aanchor = task_of(native[0]), task_of(actual[0])
                    native_anchors[nanchor] += 1; actual_anchors[aanchor] += 1
                    i = e['ifa_stats']; selected = i['selected_old_tasks']
                    assert len(selected) == len(e['weighted_ifa'])
                    es['batches'] += 1
                    es['current_windows'] += nc
                    es['old_windows'] += e['old_count']
                    es['bc_sum'] += e['bc']; es['total_loss_sum'] += e['total_loss']
                    if e['current_bc'] is not None:
                        es['current_bc_window_sum'] += e['current_bc'] * nc
                    if e['old_bc'] is not None:
                        es['old_bc_window_sum'] += e['old_bc'] * e['old_count']
                    es['weighted_ifa_batch_sum'] += sum(e['weighted_ifa'])
                    es['grad_clip_exceeded_batches'] += int(e['gradient_norm'] > 100)
                    if selected:
                        assert task in (7, 8) and selected == [task - 1], (task, selected)
                        es['ifa_batches'] += 1
                        es['ifa_window_terms'] += nc * len(selected)
                        es['hinge_window_sum'] += i['ifa_hinge_mean'] * nc * len(selected)
                        es['active_window_sum'] += i['ifa_hinge_active_fraction'] * nc * len(selected)
                        native_ifa_anchors[nanchor] += 1; actual_ifa_anchors[aanchor] += 1
                        used_language[aanchor].add(actual[0])
                        pair_counter.update(selected)
                        bucket = anchor_buckets[aanchor]
                        bucket['batches'] += 1; bucket['window_terms'] += nc
                        bucket['weighted_ifa_sum'] += sum(e['weighted_ifa'])
                        bucket['hinge_window_sum'] += i['ifa_hinge_mean'] * nc
                        bucket['active_window_sum'] += i['ifa_hinge_active_fraction'] * nc
                        es['ifa_batches_without_selected_old_in_batch'] += int(selected[0] not in e['source_tasks'])
            assert es['batches'] == (len(order) + 9) // 10
            assert {str(k): v for k, v in ec.items()} == sample['actual_counts']
            total.update(ec); per_id.update(ic); s.update(es)
            for source, ids in sorted(group_ids.items()):
                unique = sum(ic[x] > 0 for x in ids)
                per_epoch_coverage[source].append(unique)
                coverage_rows.append(dict(stage=task, epoch=epoch, source_task=source,
                    stored_windows=len(ids), draws=ec[source], unique_seen=unique,
                    zero_exposure=len(ids)-unique, duplicate_draws=ec[source]-unique))
            epoch_rows.append(dict(stage=task, epoch=epoch, **dict(es)))
        assert {str(k): v for k, v in total.items()} == routes['actual_counts'][str(task)]
        route = routes['routes'][str(task)]
        for source, ids in sorted(group_ids.items()):
            n = total[source]; native_n = len(ids) * 100
            row = dict(stage=task, source_task=source, role='current' if source == task else 'replay',
                p0_old_only=route['p0'].get(str(source)), p_old_only=route['p'].get(str(source)),
                cosine=route['cosines'].get(str(source)),
                actual_share_old_only=n/s['old_windows'] if source < task else None,
                actual_share_all_windows=n/(s['old_windows']+s['current_windows']),
                actual_draws=n, native_schedule_draws=native_n,
                change_percent=100*(n/native_n-1), stored_windows=len(ids),
                whole_stage_unique=sum(per_id[x] > 0 for x in ids),
                per_epoch_unique_mean=np.mean(per_epoch_coverage[source]),
                per_epoch_unique_min=min(per_epoch_coverage[source]),
                per_epoch_unique_max=max(per_epoch_coverage[source]),
                stage_draws_per_window_min=min(per_id[x] for x in ids),
                stage_draws_per_window_max=max(per_id[x] for x in ids))
            replay_rows.append(row)
        for anchor, b in sorted(anchor_buckets.items()):
            anchor_rows.append(dict(stage=task, anchor_task=anchor,
                native_schedule_ifa_anchor_batches=native_ifa_anchors[anchor],
                actual_ifa_anchor_batches=b['batches'],
                actual_ifa_current_window_terms=b['window_terms'],
                mean_hinge=b['hinge_window_sum']/b['window_terms'],
                active_fraction=b['active_window_sum']/b['window_terms'],
                mean_weighted_ifa_per_ifa_batch=b['weighted_ifa_sum']/b['batches'],
                anchor_equals_negative_task=(anchor == task-1)))
        for source, used in sorted(used_language.items()):
            ref = torch.load(features / str(source) / '0.pt', map_location='cpu')['feature'][0, 0, 0, :]
            exact, largest = 0, 0.
            for identity in sorted(used):
                vec = torch.load(source_to_path[identity], map_location='cpu')['feature'][0, 0, 0, :]
                exact += int(torch.equal(vec, ref))
                largest = max(largest, float((vec-ref).abs().max()))
            language_checks.append(dict(stage=task, source_task=source, unique_anchors=len(used),
                exactly_equal_to_task_0_pt=exact, max_absolute_difference=largest))
        stages[task] = dict(totals=dict(s), native_anchor_batches=dict(native_anchors),
            actual_anchor_batches=dict(actual_anchors), native_ifa_anchor_batches=dict(native_ifa_anchors),
            actual_ifa_anchor_batches=dict(actual_ifa_anchors), selected_old_task_calls=dict(pair_counter),
            mean_hinge_per_current_window_term=s['hinge_window_sum']/s['ifa_window_terms'] if s['ifa_window_terms'] else 0,
            active_fraction_per_current_window_term=s['active_window_sum']/s['ifa_window_terms'] if s['ifa_window_terms'] else 0,
            mean_ifa_added_per_all_batch=s['weighted_ifa_batch_sum']/s['batches'])
        print('Audited stage', task, 'batches', s['batches'], flush=True)
    baseline_curve = root / 'runs/final_paper_text_faithful/ifa/seed42/run_001/paper_repro_train_curve.csv'
    bc = list(csv.DictReader(baseline_curve.open()))
    vc = list(csv.DictReader((report / 'seed42/training_log.csv').open()))
    assert len(bc) == len(vc) == 400
    curve_rows = []
    for b, v in zip(bc, vc):
        assert (b['stage'], b['epoch']) == (v['stage'], v['epoch'])
        assert b['batches'] == v['steps'] and float(b['lr_end']) == float(v['lr_end'])
        curve_rows.append(dict(stage=int(b['stage']), epoch=int(b['epoch']),
            baseline_total_loss=float(b['mean_bc_loss']), v1_total_loss=float(v['total_loss']),
            delta=float(v['total_loss'])-float(b['mean_bc_loss']),
            lr_end=float(b['lr_end']), updates=int(b['batches'])))
    provenance.extend(dict(path=str(p.relative_to(root)), size=p.stat().st_size, sha256=sha(p)) for p in
        [baseline_curve, report/'ROUTING_SUMMARY.json', report/'seed42/training_log.csv',
         root/'v1_object_tsr/native_observe.py', root/'v1_object_tsr/routing.py',
         root/'libero/lifelong/algos/mlr.py'])
    csvwrite(out/'SAMPLING_EXPOSURE.csv', replay_rows)
    csvwrite(out/'EPOCH_COVERAGE.csv', coverage_rows)
    # Missing counters in zero-IFA stages are filled with zero.
    keys = list(dict.fromkeys(k for row in epoch_rows for k in row))
    csvwrite(out/'EPOCH_EVENT_SUMMARY.csv', [{k: row.get(k, 0) for k in keys} for row in epoch_rows])
    csvwrite(out/'IFA_ANCHOR_EXPOSURE.csv', anchor_rows)
    csvwrite(out/'BASELINE_LOSS_COMPARISON.csv', curve_rows)
    write(out/'LANGUAGE_ANCHOR_CHECK.json', language_checks)
    write(out/'INPUT_PROVENANCE.json', provenance)
    write(out/'AUDIT.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        type='POSTHOC_READ_ONLY_LOG_AUDIT', no_model_construction=True, no_training=True,
        no_forward_backward=True, formal_training_resumed=False, baseline_sampler='RECONSTRUCTED_NOT_BASELINE_LOGGED',
        baseline_loss_column_mean_bc_loss_is_total_bc_plus_ifa=True,
        baseline_per_batch_ifa_logs_available=False, trajectory_ids_available=False,
        validation=dict(validation), stages=stages, language_anchor_checks=language_checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
