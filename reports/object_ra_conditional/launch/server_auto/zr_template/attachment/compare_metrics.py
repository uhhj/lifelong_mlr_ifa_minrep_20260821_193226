"""Offline exact metric comparison for the locked seed43 repeat1 parent.

No training, model loading, simulation, or automatic seed extension. This
validates result-file structure and lineage fields, not the claimed GPU audit.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple

TASKS = range(6, 10)
LOCKED = {'AUC': Fraction(151, 192), 'FWT': Fraction(7, 10),
          'NBT': Fraction(-61, 360), 'OldFinal': Fraction(13, 15),
          'FinalAll': Fraction(63, 80)}


def metrics(rates: Dict[Tuple[int, int], Fraction]) -> Dict[str, Fraction]:
    expected = {(s, t) for s in TASKS for t in range(6, s + 1)}
    if set(rates) != expected:
        raise ValueError('Incomplete, extra, or wrong-stage success-rate matrix')
    if any(v < 0 or v > 1 for v in rates.values()):
        raise ValueError('Rates must be in [0,1]')
    return {
        'AUC': sum(sum(rates[s, t] for s in range(t, 10)) / (10-t) for t in TASKS) / 4,
        'FWT': sum(rates[t, t] for t in TASKS) / 4,
        'NBT': sum(sum(rates[t, t]-rates[s, t] for s in range(t+1, 10)) / (9-t) for t in range(6,9)) / 3,
        'OldFinal': sum(rates[9, t] for t in range(6, 9)) / 3,
        'FinalAll': sum(rates[9, t] for t in TASKS) / 4,
    }


def _sha_field(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Missing or malformed SHA256: ' + field)
    return value


def load_records(folder: Path, is_new: bool):
    expected_files = {'trials_stage%d_task%d.jsonl' % (s,t) for s in TASKS for t in range(6,s+1)}
    actual_files = {p.name for p in folder.glob('trials*.jsonl')}
    if actual_files != expected_files:
        raise ValueError('Must contain exactly ten formal stage/task files')
    rates, records, hashes = {}, {}, {}
    stage_hashes, stage_modal = {}, {}
    for s in TASKS:
        for t in range(6, s + 1):
            path = folder / ('trials_stage%d_task%d.jsonl' % (s,t))
            raw = path.read_bytes()
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
            if len(rows) != 20 or {r.get('trial') for r in rows} != set(range(20)):
                raise ValueError('Wrong/duplicate/missing trials: ' + path.name)
            if {r.get('initial_state_index') for r in rows} != set(range(20)):
                raise ValueError('Initial-state coverage is not 0..19')
            for r in rows:
                if (r.get('seed'), r.get('stage'), r.get('task')) != (43, s, t):
                    raise ValueError('Wrong seed/stage/task')
                if type(r.get('success')) is not bool:
                    raise ValueError('Success must be a recorded Boolean')
                if r.get('arm') != ('coreset_ifa_zrv2' if is_new else 'coreset'):
                    raise ValueError('Wrong arm; do not relabel prior outputs')
                if is_new:
                    if r.get('parent_epoch') != 99 or r.get('calibration_epoch') != 9:
                        raise ValueError('Wrong parent/calibration endpoint')
                    parent_cp = _sha_field(r.get('parent_checkpoint_sha256'), 'parent_checkpoint_sha256')
                else:
                    if r.get('epoch') != 99:
                        raise ValueError('Baseline is not the final ep99 model')
                    parent_cp = _sha_field(r.get('checkpoint_sha256'), 'checkpoint_sha256')
                _sha_field(r.get('checkpoint_sha256'), 'checkpoint_sha256')
                _sha_field(r.get('initial_state_sha256'), 'initial_state_sha256')
                modal = _sha_field(r.get('modal_sha256'), 'modal_sha256')
                if s in stage_hashes and stage_hashes[s] != parent_cp:
                    raise ValueError('Stage uses inconsistent public checkpoints')
                if s in stage_modal and stage_modal[s] != modal:
                    raise ValueError('Stage uses inconsistent front-end contracts')
                stage_hashes[s], stage_modal[s] = parent_cp, modal
                records[s,t,r['trial']] = r
            rates[s,t] = Fraction(sum(r['success'] for r in rows), 20)
    return rates, records, hashes


def compare(baseline: Path, new: Path):
    br, b, bh = load_records(baseline, False)
    nr, n, nh = load_records(new, True)
    bm, nm = metrics(br), metrics(nr)
    if bm != LOCKED:
        raise ValueError('Baseline differs from locked seed43 repeat1; stop, do not substitute')
    for key in b:
        for field in ('initial_state_index', 'initial_state_sha256', 'modal_sha256'):
            if b[key][field] != n[key][field]:
                raise ValueError('Mismatched paired input identity: ' + field)
        if b[key]['checkpoint_sha256'] != n[key]['parent_checkpoint_sha256']:
            raise ValueError('Calibration did not use the actual compared parent model')
    checks = {'AUC': nm['AUC'] > bm['AUC'], 'FWT': nm['FWT'] >= bm['FWT'],
              'NBT': nm['NBT'] < bm['NBT'], 'OldFinal': nm['OldFinal'] >= bm['OldFinal']}
    pack = lambda m: {k: {'fraction': str(v), 'percent': float(100*v)} for k,v in m.items()}
    return dict(status='OFFLINE_RECORD_COMPARISON_COMPLETE', baseline=pack(bm), new=pack(nm),
                delta_pp=pack({k:nm[k]-bm[k] for k in bm}), criteria=checks,
                metric_gate=all(checks.values()), engineering_acceptance='NOT_CHECKED_BY_THIS_TOOL',
                auto_extend=False, baseline_files_sha256=bh, new_files_sha256=nh,
                parent_id='seed43_repeat1_Coreset_IFA',
                note='Metric gate alone is not structural/GPU/protocol acceptance or statistical significance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--new-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing report; choose a new output name')
    out = compare(args.baseline_dir, args.new_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': out['status'], 'metric_gate': out['metric_gate'],
                      'auto_extend': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
