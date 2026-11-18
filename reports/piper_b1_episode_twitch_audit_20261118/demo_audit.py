from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json
import sys

import cv2
import numpy as np


def load_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def show(name, **values):
    print(name, json.dumps(values, ensure_ascii=False, separators=(',', ':')))


def stats(values, unit=1):
    if not len(values):
        return None
    numbers = np.asarray(values) / unit
    return {k: round(float(v), 4) for k, v in zip(('min','median','p95','p99','max'), np.percentile(numbers, (0,50,95,99,100)))}


def audit(episode, run, package):
    sys.path.insert(0, str(package))
    from timeline import Timeline

    rows = load_rows(episode/'alignment.jsonl')
    report = json.loads((episode/'episode_summary.json').read_text())
    stamps = np.array([r['sample_monotonic_ns'] for r in rows], dtype=np.int64)
    event_rows = load_rows(run/'can_timeline.jsonl')
    events = iter(event_rows)
    event = next(events, None)
    model = Timeline()
    comparison = Counter()
    for row, stamp in zip(rows, stamps.tolist()):
        while event and event['mono_ns'] <= stamp+100_000_000:
            model.ingest(event)
            event = next(events, None)
        comparison['feedback_identity'] += model.observation_at(stamp) == row['feedback']
        comparison['target_identity'] += model.command_at(stamp) == row['target']
    indices = [f'{Decimal(ns)/Decimal(1_000_000_000):.6f}' for ns in stamps.tolist()]
    base = episode/'episode0'
    inputs = {'overhead': base/'camera/color/overhead', 'wrist': base/'camera/color/wrist',
              'target': base/'array/float32/follower_target', 'feedback': base/'array/float32/follower_feedback'}
    bad = Counter()
    for name, p in inputs.items():
        files = {f.stem:f for f in p.iterdir() if f.stem.replace('.','').isdecimal()}
        show('FILES', stream=name, count=len(files), missing=len(set(indices)-files.keys()), extra=len(files.keys()-set(indices)))
        prev = None
        for key,row in zip(indices,rows):
            if key not in files: continue
            f = files[key]
            if name in ('overhead','wrist'):
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None or img.shape != (480,640,3):
                    bad[f'{name}_invalid'] += 1
                    continue
                if prev is not None and np.array_equal(prev,img):
                    bad[f'{name}_identical_neighbors'] += 1
                prev = img
            else:
                arr = np.load(f, allow_pickle=False)['data'].ravel()
                ref = row[name]
                goal = [] if ref is None else [*ref['joints_rad'],ref['gripper_mm']]
                if arr.shape!=(len(goal),) or not np.allclose(arr,goal,rtol=1e-7,atol=1e-9):
                    bad[f'{name}_mismatch'] += 1
    feedback_ages = [r['sample_monotonic_ns']-r['feedback']['sample_monotonic_ns'] for r in rows if r['feedback']]
    targets = [(i,r['target']) for i,r in enumerate(rows) if r['target']]
    first = targets[0][0]
    skew = [abs(r['cameras']['overhead']['monotonic_ns']-r['cameras']['wrist']['monotonic_ns']) for r in rows]
    missing = [i for i,r in enumerate(rows) if r['target'] is None]
    intervals=np.diff(stamps)
    show('QUALITY', task=report['task'], outcome=report['outcome'], samples=len(rows), duration_s=(stamps[-1]-stamps[0])/1e9,
         effective_hz=round((len(rows)-1)*1e9/(stamps[-1]-stamps[0]),4), intervals_ms=stats(intervals,1e6),
         camera_skew_ms=stats(skew,1e6), feedback_age_ms=stats(feedback_ages,1e6),
         first_valid_index=first, null_targets=len(missing), invalid_after_first=sum(i>first for i in missing),
         widths_mm=[min(r['feedback']['gripper_mm'] for r in rows),max(r['feedback']['gripper_mm'] for r in rows)],
         noncausal_targets=sum(t['sample_monotonic_ns']>rows[i]['sample_monotonic_ns'] for i,t in targets),
         replay=dict(comparison), problems=dict(bad))
    for offset in (0,first,106,212,318,424,len(rows)-1):
        row=rows[offset]
        show('TIMEMARK', i=offset, from_start_s=round((stamps[offset]-stamps[0])/1e9,3),
             target=bool(row['target']), measured_grip_mm=row['feedback']['gripper_mm'])
    return inputs, indices, stamps

if __name__ == '__main__':
    episode,run,package=map(Path,sys.argv[1:])
    audit(episode,run,package)
