"""Offline audit of B1 execution, synchronized model frames and archived USB measurements.

Usage: python -B audit.py WORK STATION_SNAPSHOT SESSION [SESSION ...]
Reads existing files only. Standard output contains aggregate metrics and hashes.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def stats(values):
    return dict(zip(('min', 'median', 'p95', 'max'),
                    np.round(np.percentile(values, [0, 50, 95, 100]), 3).tolist()))


def audit(work, snapshot, name):
    folder = work / 'policy_runs' / name
    log = folder.with_suffix('.jsonl')
    index = folder / 'inputs.jsonl'
    rows = records(log)
    samples = records(index)
    by_stamp = {r['stamp']: r for r in samples}
    header = rows[0]
    profile = header['policy']
    bundle = profile['alignment']
    assert hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == profile['sha256']
    code_hash = {n: sha(snapshot / n) for n in ('b1_alignment_gate.py', 'wrist_cue.py')}
    assert all(bundle['station_files'][n] == h for n, h in code_hash.items())
    from b1_alignment_gate import OVERHEAD_TOLERANCE_MM, wrist_block, wrist_centered
    from wrist_cue import signed_cue
    start = next(r['mono_ns'] for r in rows if r['event'] == 'gripper_stable')
    stop = next(r for r in rows if r['event'] == 'pika_stop_trace')
    when = stop['stop_requested_ns']
    secs = lambda n: round((n-start)/1e9, 5)
    offset = header['wall_ns'] - header['mono_ns']
    local_start = datetime.fromtimestamp((start+offset)/1e9, timezone(timedelta(hours=8)))
    sent = [r for r in rows if r.get('mode') == 'execute' and r.get('sent')]
    streak = longest = 0
    prev = None
    boxes = centered_count = within = late_blank = 0
    onset = None
    aligned_times = []
    within_intent = intent_count = 0
    heads = []
    sides = Counter()
    confirmed = Counter()
    distal_votes = Counter()
    dists = []
    track = []
    sent_delta = []
    skew = []
    age = []
    for r in sent:
        stamp = r['stamp']
        entry = by_stamp[stamp]
        assert entry['mode_at_query'] == 'execute' and entry['images'] == r['images']
        assert np.allclose(entry['p'], r['p']) and np.allclose(entry['q'], r['q'])
        assert r['full'] and len(set(entry['history'])) == 8
        a = r['grasp_alignment']
        assert a['stamp'] == stamp
        image = cv2.imread(str(folder / r['images']['model']))
        assert image.shape == (480, 1280, 3)
        wrist = image[:, 640:].copy()
        box = wrist_block(wrist)
        assert (list(box) if box is not None else None) == a['wrist_box']
        centered = wrist_centered(box, r['q'][6])
        inside = a['distance_mm'] <= OVERHEAD_TOLERANCE_MM
        assert (inside and centered) == a['aligned']
        boxes += int(box is not None)
        centered_count += int(centered)
        within += int(inside)
        close = r['close_intent']
        if close:
            intent_count += 1
            within_intent += int(inside)
            if onset is None:
                onset = secs(stamp)
        if close and inside:
            late_blank += int(box is None)
            distal_votes[str(signed_cue(wrist))] += 1
        if a['aligned']:
            streak = (streak if prev is not None and stamp-prev <= 50_000_000 else 0) + 1
            aligned_times.append(secs(stamp))
        else:
            streak = 0
        longest = max(longest, streak)
        prev = stamp
        dists.append(a['distance_mm'])
        sides[str(a['wrist_side'])] += 1
        confirmed[str(a['confirmed_side'])] += 1
        plan = entry['plan']
        if plan['chunk_stamp'] == stamp and plan['offset'] == 0:
            opts = plan.get('alignment', {})
            if 'revised_xy_mm' in opts:
                heads.append(dict(s=secs(stamp), mode=a['confirmed_side'], overhead_mm=round(dists[-1], 3),
                                  original_mm=round(float(np.linalg.norm(opts['baseline_xy_mm'])), 3),
                                  revised_mm=round(float(np.linalg.norm(opts['revised_xy_mm'])), 3)))
        track.append(np.abs(np.rad2deg(np.array(r['sent'][:6])-r['measured_at_send'][:6])))
        sent_delta.append(np.max(np.abs(np.rad2deg(np.array(r['p'][:6])-r['sent'][:6]))))
        skew.append(entry['skew_ns']/1e6)
        age.append((r['send_started_ns']-stamp)/1e6)

    tail = next((e for e in samples if e['mode_at_query'] == 'execute' and e['stamp'] > sent[-1]['stamp']), None)
    timing = None
    if tail:
        predecessor = by_stamp[sent[-1]['stamp']]
        stamp = tail['stamp']
        timing = dict(stamp_to_stop_ms=round((when-stamp)/1e6, 3),
                      image_pair_gap_ms=round((stamp-predecessor['stamp'])/1e6, 3),
                      image_to_prediction_ms=round((tail['predicted_ns']-stamp)/1e6, 3),
                      prediction_to_stop_ms=round((when-tail['predicted_ns'])/1e6, 3),
                      distinct_history=len(set(tail['history'])), plan_ready=tail['plan']['ready'],
                      pair_skew_ms=round(tail['skew_ns']/1e6, 3),
                      camera_sequence_steps={k:v['sequence']-predecessor['cameras'][k]['sequence']
                                             for k,v in tail['cameras'].items()},
                      image_to_file_mtime_ms=round(((folder/tail['images']['model']).stat().st_mtime_ns-offset-stamp)/1e6, 3),
                      last_ten_sent_image_to_file_mtime_ms=stats([
                          ((folder/r['images']['model']).stat().st_mtime_ns-offset-r['stamp'])/1e6 for r in sent[-10:]]))
    frames = stop['frames']
    zeros = [i for i, f in enumerate(frames) if f['width_mm'] == 0]
    zero_trace = None
    if zeros:
        indices = [min(zeros)-1, *zeros, max(zeros)+1]
        assert indices[0] >= 0 and indices[-1] < len(frames)
        zero_trace = []
        for i in indices:
            f = frames[i]
            payload = json.loads(f['usb_frame'])
            motor = payload['motor']
            assert float(payload['motorstatus']['Voltage']) == f['voltage_v']
            zero_trace.append(dict(relative_to_stop_ms=round((f['rx_ns']-when)/1e6, 3),
                                   width_mm=round(f['width_mm'], 3), position=motor['Position'],
                                   current_ma=motor['Current'], speed=motor['Speed'], volts=f['voltage_v'],
                                   status=payload['motorstatus']['Status']))
    return dict(session=name, execution_start_beijing=local_start.isoformat(timespec='milliseconds'),
                profile_sha256=profile['sha256'], log_sha256=sha(log), inputs_sha256=sha(index),
                vision_code_sha256=code_hash, stop_after_s=secs(when), end=rows[-1]['detail'],
                deadline_s=profile['duration'], dispatched=len(sent), close_intent_ticks=intent_count,
                first_close_intent_s=onset, phases=dict(Counter(r['gripper_phase'] for r in sent)),
                usb_width_command_count=sum(r.get('pika_target_mm') is not None for r in sent),
                interrupted_dispatches=sum(r['event']=='send_interrupted' for r in rows),
                aligned_s=aligned_times, longest_aligned_streak=longest,
                projected_error_mm=dict(first=round(dists[0], 3), minimum=round(min(dists), 3), last=round(dists[-1], 3)),
                overhead_within_ticks=within, overhead_within_and_intent_ticks=within_intent,
                wrist_box_ticks=boxes, wrist_centered_ticks=centered_count,
                missing_wrist_box_during_intent_within=late_blank, standalone_side_during_intent_within=dict(distal_votes),
                camera_side_counts=dict(sides), confirmed_side_counts=dict(confirmed), wrist_modified_heads=heads,
                six_axis_pred_to_sent_max_deg=stats(sent_delta), six_axis_sent_to_same_time_feedback_max_deg=stats(np.max(track,axis=1)),
                reductions_ticks=sum(bool(r.get('tracking_reductions')) for r in sent), pair_skew_ms=stats(skew),
                image_to_dispatch_ms=stats(age), first_undispatched=timing, stop_window_zero_width_count=len(zeros),
                stop_window_zero_neighborhood=zero_trace)


if __name__ == '__main__':
    work, snapshot = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(snapshot))
    cv2.setNumThreads(1)
    print(json.dumps({'trials':[audit(work, snapshot, name) for name in sys.argv[3:]]}, ensure_ascii=True))
