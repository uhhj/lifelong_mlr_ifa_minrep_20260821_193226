"""Check exported annotation integrity; does not certify visual judgments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .export_ai_labels import atomic_new, json_text


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(report, pack):
    completion = json.loads((report / 'COMPLETE.json').read_text(encoding='utf-8'))
    for name, expected in completion['outputs'].items():
        assert digest(report / name) == expected, name
    with (report / 'AI_LABELS.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 450 and len({r['pair_id'] for r in rows}) == 450
    assert [int(r['index']) for r in rows] == list(range(450))
    trajectory_sets = {role: set() for role in ('calibration', 'validation')}
    for row in rows:
        assert row['label_source'] == 'AI_VISUAL_ANNOTATION'
        assert row['human_verified'] == 'False'
        assert row['note'] and row['confidence']
        assert row['protocol_sha256'] == completion['bindings']['protocol_sha256']
        assert digest(pack / 'annotation/images' / (row['pair_id'] + '.png')) == row['image_sha256']
        assert row['label'] == ('' if row['decision'] == 'U' else row['decision'])
        for key in ('id_a', 'id_b'):
            trajectory_sets[row['role']].add(row[key].rsplit('/t', 1)[0])
    assert not (trajectory_sets['calibration'] & trajectory_sets['validation'])
    for role in ('calibration', 'validation'):
        with (report / (role.upper() + '_AI_LABELS.csv')).open(encoding='utf-8') as f:
            binary = list(csv.DictReader(f))
        assert binary == [r for r in rows if r['role'] == role and r['decision'] != 'U']
    # The historical blank labels and images are immutable sources, not AI output slots.
    with (pack / 'ANNOTATION_MANIFEST.csv').open(encoding='utf-8') as f:
        assert all(r['label'] == '' for r in csv.DictReader(f))
    result = dict(status='PASS', audit_kind='artifact integrity, not independent label accuracy',
                  pairs_verified=len(rows), image_sha_verified=450,
                  trajectory_overlap=0, human_verified=False,
                  formal_training_resumed=False)
    atomic_new(report / 'INTEGRITY_CHECK.json', json_text(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--pack', type=Path, required=True)
    args = p.parse_args()
    print(json_text(check(args.report, args.pack)))
