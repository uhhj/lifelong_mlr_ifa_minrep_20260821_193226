"""One frozen base-only calibration followed by disjoint-trajectory validation.

Labels are AI annotations, never human ground truth. This wrapper reuses the
original script's numerical rules unchanged. No incremental outcomes are read.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from abcr_pair_validation_b_v1 import original_experiment_b as original


def metrics(rows, weights=None):
    if weights is None:
        weights = np.ones(len(rows))
    counts = dict(tp=0., fp=0., tn=0., fn=0.)
    for r, weight in zip(rows, weights):
        key = ('tp' if r['predicted_match'] else 'fn') if r['label'] == 1 else ('fp' if r['predicted_match'] else 'tn')
        counts[key] += float(weight)
    def ratio(a, b):
        return a/b if b else None
    return dict(**counts, precision=ratio(counts['tp'], counts['tp']+counts['fp']),
                recall=ratio(counts['tp'], counts['tp']+counts['fn']),
                fpr=ratio(counts['fp'], counts['fp']+counts['tn']),
                accuracy=ratio(counts['tp']+counts['tn'], sum(counts.values())))


def trajectory_bootstrap(rows, repeats=2000, seed=20261014):
    """Resample trajectories within task; pair weight = endpoint multiplicities.

    Accounts for repeated endpoints, unlike treating pairs as independent.
    Sparse strata and zero observed errors can still make intervals optimistic.
    """
    groups = {}
    for r in rows:
        for side in ('a', 'b'):
            groups.setdefault(r['task_'+side], set()).add(r['traj_'+side])
    groups = {k: sorted(v) for k, v in groups.items()}
    rng = np.random.RandomState(seed)
    samples = {k: [] for k in ('precision', 'recall', 'fpr', 'accuracy')}
    for _ in range(repeats):
        multiplicities = {}
        for task, trajectories in sorted(groups.items()):
            count = rng.multinomial(len(trajectories), np.full(len(trajectories), 1/len(trajectories)))
            multiplicities.update(zip(trajectories, count))
        m = metrics(rows, [multiplicities[r['traj_a']]*multiplicities[r['traj_b']] for r in rows])
        for key in samples:
            if m[key] is not None:
                samples[key].append(m[key])
    return dict(method='within-task trajectory bootstrap; product of endpoint multiplicities',
                repeats=repeats, seed=seed,
                limitations='Descriptive intervals for selected physical hard pairs; sparse/zero cells may be degenerate; no population coverage guarantee.',
                intervals={k: dict(ci95=np.quantile(v, [.025,.975]).tolist() if v else None, valid_repeats=len(v)) for k,v in samples.items()})


def atomic_csv(path, rows):
    temp = path.with_name(path.name+'.tmp')
    with open(temp, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True)
    args = ap.parse_args()
    out, labels = args.out, args.labels
    export = json.loads((out/'FEATURE_EXPORT_COMPLETE.json').read_text())
    assert export['status'] == 'PASS'
    for name, digest in export['outputs'].items():
        assert original.sha256(out/name) == digest, 'FEATURE_OUTPUT_HASH_MISMATCH'
    complete = json.loads((labels/'COMPLETE.json').read_text(encoding='utf-8'))
    assert complete['label_source'] == 'AI_VISUAL_ANNOTATION' and complete['human_verified'] is False
    for name, digest in complete['outputs'].items():
        assert original.sha256(labels/name) == digest, 'LABEL_OUTPUT_HASH_MISMATCH'
    assert original.sha256(original.__file__) == 'e5905b95b34f8d27041a8181313f9fc3cd6cf7ca6df814d2433437220869b853'
    label_sha = {name: original.sha256(labels/name) for name in ['CALIBRATION_AI_LABELS.csv', 'VALIDATION_AI_LABELS.csv', 'AI_LABELS.csv']}
    bindings = dict(feature_complete_sha256=original.sha256(out/'FEATURE_EXPORT_COMPLETE.json'),
                    label_sha256=label_sha, evaluator_sha256=original.sha256(__file__),
                    amendment_sha256=complete['bindings']['protocol_sha256'])
    protocol = dict(schema='ABCR_B_V2_CPU_BASE_VALIDATION', bindings=bindings,
                    physical=original.DEFAULTS, q=3, min_calibration_positive=20, min_calibration_negative=40,
                    action_scale_scope='Unique calibration-role base anchors only, excludes validation actions',
                    label_source='AI_VISUAL_ANNOTATION', human_verified=False,
                    data_role='calibration demo0..24 / validation demo25..49, all tasks0..5',
                    validation_gate='Original protocol gives qualitative poor-state-comparability stop, no numerical precision/recall cutoff; do not invent a post-hoc PASS cutoff.',
                    bootstrap=dict(repeats=2000, seed=20261014, unit='trajectory within task'),
                    formal_training_resumed=False)
    if (out/'RUN_PROTOCOL.json').exists():
        assert json.loads((out/'RUN_PROTOCOL.json').read_text()) == protocol, 'RESUME_BINDING_MISMATCH'
    else:
        original.dump_json(out/'RUN_PROTOCOL.json', protocol)
    fr_path = out/'frozen_original_rules.json'
    if not fr_path.exists():
        # No validation scores are computed before this immutable file exists.
        original.cmd_calibrate(SimpleNamespace(records=str(out/'calibration_records.npz'),
            pairs=str(labels/'CALIBRATION_AI_LABELS.csv'), encoder_sha256=export['binding']['checkpoint_sha256'],
            max_base_task=5, q=3, pos_m=.03, rot_deg=15., grip_frac=.10, velocity_mps=.05,
            angular_velocity_radps=.35, min_pos=20, min_neg=40, negative_quantile=.95,
            motion_share=.5, motion_bif=1.5, out=str(fr_path)))
    fr = json.loads(fr_path.read_text())
    assert fr['calibration']['records_sha256'] == original.sha256(out/'calibration_records.npz')
    assert fr['calibration']['labels_csv_sha256'] == label_sha['CALIBRATION_AI_LABELS.csv']
    if not (out/'CALIBRATION_SEAL.json').exists():
        original.dump_json(out/'CALIBRATION_SEAL.json', dict(bindings=bindings,
            frozen_sha256=original.sha256(fr_path), label_source='AI_VISUAL_ANNOTATION', human_verified=False,
            validation_scores_read=False, formal_training_resumed=False))
    else:
        seal = json.loads((out/'CALIBRATION_SEAL.json').read_text())
        assert seal['bindings'] == bindings and seal['frozen_sha256'] == original.sha256(fr_path)
    all_labels = original.read_csv(labels/'AI_LABELS.csv')
    output_rows, result, task_rows = [], {}, []
    trajectories = {}
    numeric_errors = dict(cosine_max_abs=0., physical_max_abs=0., motion_max_abs=0.)
    for role in ('calibration', 'validation'):
        d = original.load_records(out/(role+'_records.npz'))
        trajectories[role] = set(d['traj_ids'].tolist())
        source_rows = [r for r in all_labels if r['role'] == role]
        pairs = [(d['index'][r['id_a']], d['index'][r['id_b']], None) for r in source_rows]
        assert all(d['alignment_verified'][i] and d['alignment_verified'][j] for i,j,_ in pairs)
        desc = original.pair_descriptors(d, pairs, .03, 15., .10)
        valid, motion, conflict = original.motion_pair(d, pairs, 3, np.array(fr['action']['motion_scale_base_IQR_floor_0p1']))
        role_rows = []
        for n, ((i,j,_), label_row) in enumerate(zip(pairs, source_rows)):
            # Independent scalar arithmetic, not a second call to the original checker.
            cos = []
            for key in ('agent_unmod', 'eih_unmod'):
                a,b = d[key][i].astype(float), d[key][j].astype(float)
                cos.append(float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))))
            qa,qb = d['eef_quat_xyzw'][i],d['eef_quat_xyzw'][j]
            rotation = 2*np.arccos(np.clip(abs(np.dot(qa,qb))/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1))
            distance = max(np.linalg.norm(d['eef_pos_m'][i]-d['eef_pos_m'][j])/.03,
                           rotation/np.deg2rad(15), abs(d['gripper_open_fraction'][i]-d['gripper_open_fraction'][j])/.10)
            diff = (d['future_actions'][i,:,:6].astype(float)-d['future_actions'][j,:,:6].astype(float))/fr['action']['motion_scale_base_IQR_floor_0p1']
            # Original subtracts stored float32 actions before division; preserve and report its rounding.
            numeric_errors['motion_max_abs'] = max(numeric_errors['motion_max_abs'], abs(float(np.sqrt(np.mean(diff**2)))-float(motion[n])))
            numeric_errors['cosine_max_abs'] = max(numeric_errors['cosine_max_abs'], abs(min(cos)-desc['sv'][n]))
            numeric_errors['physical_max_abs'] = max(numeric_errors['physical_max_abs'], abs(distance-desc['ds'][n]))
            phys = bool(desc['ds'][n] <= 1)
            visual = bool(desc['sv'][n] >= fr['visual']['tau_v'])
            gated = phys and visual and bool(valid[n])
            category = ('SHARE' if gated and motion[n] <= .5 and not conflict[n] else
                        'BIF' if gated and (motion[n] >= 1.5 or conflict[n]) else
                        'UNCERTAIN_ACTION' if gated else 'GATE_REJECTED')
            r = dict(pair_id=label_row['pair_id'], role=role, id_a=d['ids'][i], id_b=d['ids'][j],
                task_a=int(d['task_ids'][i]), task_b=int(d['task_ids'][j]), traj_a=str(d['traj_ids'][i]), traj_b=str(d['traj_ids'][j]),
                label=int(label_row['label']) if label_row['label'] in ('0','1') else 'U',
                label_source='AI_VISUAL_ANNOTATION', human_verified=False,
                physical_distance=float(desc['ds'][n]), agent_cos=float(desc['agent_cos'][n]), eih_cos=float(desc['eih_cos'][n]),
                min_cos=float(desc['sv'][n]), physical_pass=phys, visual_pass=visual, predicted_match=phys and visual,
                future_valid=bool(valid[n]), motion_distance=float(motion[n]), gripper_conflict=bool(conflict[n]), category=category)
            output_rows.append(r)
            role_rows.append(r)
        labeled = [r for r in role_rows if r['label'] != 'U']
        result[role] = dict(total=len(role_rows), labeled=len(labeled), unresolved=len(role_rows)-len(labeled),
            physical_pass=sum(r['physical_pass'] for r in role_rows), match_pass=sum(r['predicted_match'] for r in role_rows),
            metrics_relative_to_ai=metrics(labeled), trajectory_bootstrap=trajectory_bootstrap(labeled),
            categories={k:sum(r['category']==k for r in role_rows) for k in ('SHARE','BIF','UNCERTAIN_ACTION','GATE_REJECTED')},
            score_quantiles={str(label): np.quantile([r['min_cos'] for r in role_rows if r['label']==label], [0,.25,.5,.75,.95,1]).tolist()
                             for label in (0,1,'U') if any(r['label']==label for r in role_rows)})
        for a,b in sorted(set((r['task_a'],r['task_b']) for r in role_rows)):
            subset = [r for r in labeled if (r['task_a'],r['task_b'])==(a,b)]
            task_rows.append(dict(role=role,task_a=a,task_b=b,n=len(subset),**metrics(subset)))
    overlap = trajectories['calibration'] & trajectories['validation']
    assert not overlap
    assert numeric_errors['cosine_max_abs'] < 1e-12 and numeric_errors['physical_max_abs'] < 1e-10
    assert numeric_errors['motion_max_abs'] < 1e-6
    atomic_csv(out/'BASE_PAIR_SCORES.csv',output_rows)
    atomic_csv(out/'BASE_VALIDATION_BY_TASK_PAIR.csv',task_rows)
    original.dump_json(out/'INDEPENDENT_NUMERICAL_CHECK.json', dict(status='PASS', n_pairs=len(output_rows),
        errors=numeric_errors, trajectory_overlap=0, direct_hdf5_anchors=True,
        incremental_pt_alignment_verified=False, formal_training_resumed=False))
    original.dump_json(out/'BASE_VALIDATION_REPORT.json', dict(label_source='AI_VISUAL_ANNOTATION', human_verified=False,
        frozen_sha256=original.sha256(fr_path), tau_v=fr['visual']['tau_v'], roles=result,
        trajectory_counts={k:len(v) for k,v in trajectories.items()},trajectory_overlap=0,
        encoder_pretrained_on_base_demonstrations=True,
        isolation_scope='Label calibration/validation trajectories are disjoint; not held out from original encoder pretraining',
        formal_training_resumed=False))
    print(json.dumps(dict(tau_v=fr['visual']['tau_v'],results=result),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
