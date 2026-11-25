"""Measure changes after a command using subsequent, independently timestamped gripper feedback."""

import argparse
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import median


def time_label(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def mm(value):
    return round(value, 3)


def field(path):
    with path.open(encoding="utf-8") as src:
        rows = [json.loads(line) for line in src]
    begin = next(r for r in rows if r["event"] == "start")
    sent = [r for r in rows if r.get("mode") == "execute"]
    samples = {}

    def add(f):
        t = f["stamp_ns"]
        point = (f["width_mm"], f["torque_nm"], f["status"])
        assert samples.setdefault(t, point) == point

    for r in sent:
        assert r["send_started_ns"] <= r["send_finished_ns"] and 0x159 in r["sent_ids"]
        add(r["gripper_at_image"])
        add(r["gripper_control"]["feedback"])
    add(next(r for r in rows if r["event"] == "gripper_end")["last_check"]["feedback"])
    start = begin["mono_ns"]
    ticks = sorted(t for t in samples if t >= start)

    def previous(time):
        i = bisect_right(ticks, time) - 1
        assert i >= 0
        return samples[ticks[i]]

    waiting = [r for r in sent if r["gripper_control"]["phase"] == "waiting"]
    assert len({r["sent"][6] for r in waiting}) == 1
    wait = waiting[0]["gripper_control"]["mono_ns"]
    tail = [samples[t] for t in ticks if t >= wait]
    center = median(v[0] for v in tail)
    # The last uninterrupted occupancy of the final feedback band, including pre-wait samples.
    outside = [i for i, t in enumerate(ticks) if abs(samples[t][0] - center) > 0.100001]
    flat = ticks[outside[-1] + 1 if outside else 0]
    assert flat <= wait
    # Work backwards to find the final sustained closing error >=4 mm leading to the wait.
    tracking = [r for r in sent if r["gripper_control"]["mono_ns"] <= wait]
    last_below = max(i for i, r in enumerate(tracking) if r["measured_at_send"][6] - r["sent"][6] < 4)
    anchor = tracking[last_below + 1]
    a = anchor["send_finished_ns"]
    ref = previous(a)[0]
    later = []
    for dt in (100, 250, 500):
        t = a + dt * 1_000_000
        i = bisect_right(ticks, t) - 1
        assert 0 <= t - ticks[i] < 50_000_000
        later.append(mm(ref - samples[ticks[i]][0]))
    ft = [r["send_finished_ns"] for r in sent]
    assert ft == sorted(ft)
    entering_flat = bisect_right(ft, flat) - 1
    assert entering_flat >= 0

    def since(t):
        return round((t - start) / 1e9, 3)

    return {
        "start": time_label(start + rows[0]["wall_ns"] - rows[0]["mono_ns"]),
        "initial_width_mm": begin["initial"][6],
        "min_max_feedback_mm": [min(samples[t][0] for t in ticks), max(samples[t][0] for t in ticks)],
        "sustained_4mm_gap": {
            "t_s": since(a),
            "target_mm": mm(anchor["sent"][6]),
            "width_mm": ref,
            "closing_motion_after_100_250_500ms_mm": later,
        },
        "final_band": {
            "center_mm": center,
            "first_t_s": since(flat),
            "lead_before_wait_s": round((wait - flat) / 1e9, 3),
            "total_observed_s": round((ticks[-1] - flat) / 1e9, 3),
            "target_on_entry_mm": mm(sent[entering_flat]["sent"][6]),
            "min_max_width_mm": [
                min(samples[t][0] for t in ticks if t >= flat),
                max(samples[t][0] for t in ticks if t >= flat),
            ],
        },
        "waiting": {
            "first_t_s": since(wait),
            "target_mm": mm(waiting[0]["sent"][6]),
            "submissions": len(waiting),
            "torque_median_nm": round(median(v[1] for v in tail), 4),
            "unique_feedback_points": len(tail),
        },
    }


def wire(series):
    commands = series["commands"]
    start, end = series["start"], series["end"]
    points = [r for r in series["feedback"] if start <= r[0] <= end]
    assert points and all(c[2:] == [1000, 1, 0] for c in commands)
    assert all(r[3] == 0x40 for r in points)
    times = [r[0] for r in commands]
    assert times == sorted(times)
    # Use the last observed target at each feedback timestamp. A new target replaces the previous one.
    max_gap = max(b[0] - a[0] for a, b in zip(points, points[1:])) / 1e6
    longest = []
    current = []
    for r in points:
        target = commands[bisect_right(times, r[0]) - 1][1]
        if (not current or r[0] - current[-1][0] <= 20_000_000) and r[1] - target >= 4000:
            current.append(r)
        else:
            if current and (not longest or current[-1][0] - current[0][0] > longest[-1][0] - longest[0][0]):
                longest = current
            current = [r] if r[1] - target >= 4000 else []
    if current and (not longest or current[-1][0] - current[0][0] > longest[-1][0] - longest[0][0]):
        longest = current

    def width_range(seq):
        vals = [r[1] / 1000 for r in seq]
        return [min(vals), max(vals)]

    def span(seq):
        return round((seq[-1][0] - seq[0][0]) / 1e9, 3)

    interval = None
    if longest:
        interval = {
            "duration_s": span(longest),
            "width_range_mm": width_range(longest),
            "torque_median_nm": median(r[2] for r in longest) / 1000,
            "torque_min_max_nm": [min(r[2] for r in longest) / 1000, max(r[2] for r in longest) / 1000],
        }
    raw = series["feedback"]
    raw_times = [r[0] for r in raw]
    pre = [r for r in raw if start - 100_000_000 <= r[0] < start]
    assert pre and start - pre[-1][0] < 20_000_000
    width0 = pre[-1][1]
    early = [r for r in points if r[0] <= start + 250_000_000]

    def first_ms(matching):
        return next((round((r[0] - start) / 1e6, 1) for r in points if matching(r)), None)

    follow = []
    for delay in (0, 100, 250, 500, 1000):
        t = start + delay * 1_000_000
        if t > end:
            break
        cmd = commands[bisect_right(times, t) - 1]
        fb = raw[bisect_right(raw_times, t) - 1]
        assert 0 <= t - fb[0] < 20_000_000
        follow.append(
            {
                "after_ms": delay,
                "target_mm": cmd[1] / 1000,
                "width_mm": fb[1] / 1000,
                "torque_nm": fb[2] / 1000,
            }
        )
    return {
        "start": time_label(start),
        "commands": len(commands),
        "feedback": len(points),
        "max_feedback_gap_ms": round(max_gap, 2),
        "span_s": span(commands),
        "target_range_mm": width_range(commands),
        "width_range_mm": width_range(points),
        "first_last_target_mm": [commands[0][1] / 1000, commands[-1][1] / 1000],
        "policy_overlap": series["overlaps_policy"],
        "width_range_before_100ms_mm": width_range(pre),
        "first_0p3mm_closing_ms": first_ms(lambda r: width0 - r[1] >= 300),
        "first_negative_0p2Nm_ms": first_ms(lambda r: r[2] <= -200),
        "first_250ms_torque_min_max_nm": [min(r[2] for r in early) / 1000, max(r[2] for r in early) / 1000],
        "command_feedback_pairs": follow,
        "error_or_stop_or_enable_inside": sum(start <= r[0] <= end for r in series["safety"]),
        "longest_closing_gap_at_least_4mm": interval,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("can_archive", type=Path)
    parser.add_argument("field_logs", nargs="+", type=Path)
    args = parser.parse_args()
    archived = json.loads(args.can_archive.read_bytes())
    print(
        json.dumps(
            {
                "field": [field(p) for p in sorted(args.field_logs, key=lambda p: p.name)],
                "wire": [wire(s) for s in archived["series"]],
                "isolated_manual_matches": len(archived["isolated"]),
            },
            indent=2,
            allow_nan=False,
        )
    )
