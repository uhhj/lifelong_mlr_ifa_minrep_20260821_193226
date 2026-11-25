"""Audit target provenance, control status and subsequent measured effort in archived CAN windows."""

from argparse import ArgumentParser
from ast import parse, walk, FunctionDef
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from struct import Struct
from zipfile import ZipFile

FRAME = Struct("<QQIBBH8s")
GRIP = Struct(">ihBB")
JOINTS = Struct(">ii")
LOCAL = 4


def effort(points):
    values = [data[1] for _, data in points]
    return {
        "n": len(values),
        "min_median_max_counts": [min(values), median(values), max(values)],
        "at_or_below_minus_50": sum(n <= -50 for n in values),
        "at_or_below_minus_100": sum(n <= -100 for n in values),
        "at_or_below_minus_200": sum(n <= -200 for n in values),
    }


def window(data, start):
    frames = sorted(
        (t, cid, flags, payload)
        for t, _, cid, port, size, flags, payload in FRAME.iter_unpack(data)
        if port == 0 and (size == 8 or cid & 0x20000000)
    )
    targets = [
        (t, GRIP.unpack(payload)) for t, cid, flags, payload in frames if cid == 0x159 and flags & LOCAL
    ]
    assert targets[0][0] == start
    end = targets[-1][0]
    active = [r for r in frames if start <= r[0] <= end]
    raw_feedback = [
        (t, GRIP.unpack(payload)) for t, cid, flags, payload in frames if cid == 0x2A8 and not flags & LOCAL
    ]
    pre = [(t, d) for t, d in raw_feedback if start - 300_000_000 <= t < start]
    fb = [(t, d) for t, d in raw_feedback if start <= t <= end]
    early = [(t, d) for t, d in fb if t <= start + 250_000_000]

    def ms(t):
        return round((t - start) / 1e6, 3)

    initial = pre[-1][1][0]
    pulse = next((ms(t) for t, d in fb if d[1] <= -200), None)
    motion = next((ms(t) for t, d in fb if initial - d[0] >= 300), None)
    # Find the last entry into the range at or below 0.1 mm, rather than counting earlier crossings.
    above = [i for i, (_, d) in enumerate(targets) if d[0] > 100]
    low_start = (above[-1] + 1) if above else 0
    low = targets[low_start:] if low_start < len(targets) else []
    low_fb = [(t, d) for t, d in fb if low and t >= low[0][0]]
    arm = Counter(p.hex() for _, cid, flags, p in active if cid == 0x2A1 and not flags & LOCAL)
    spans = []
    for cid in (0x2A5, 0x2A6, 0x2A7):
        joint = [JOINTS.unpack(p) for _, j, flags, p in active if j == cid and not flags & LOCAL]
        spans.extend(round((max(v) - min(v)) / 1000, 3) for v in zip(*joint))
    return {
        "start": datetime.fromtimestamp(start / 1e9, timezone(timedelta(hours=8))).isoformat(
            timespec="milliseconds"
        ),
        "window_sha256": sha256(data).hexdigest(),
        "command_duration_s": round((end - start) / 1e9, 4),
        "local_targets": len(targets),
        "bus_origin_targets_0x159": sum(cid == 0x159 and not flags & LOCAL for _, cid, flags, _ in active),
        "local_target_effort_code_zero": sorted({d[1:] for _, d in targets}),
        "feedback_status_mode": sorted({d[2:] for _, d in fb}),
        "initial_width_mm": initial / 1000,
        "actual_width_range_mm": [min(d[0] for _, d in fb) / 1000, max(d[0] for _, d in fb) / 1000],
        "pre_300ms": effort(pre),
        "first_250ms": effort(early),
        "whole": effort(fb),
        "first_minus_200_counts_ms": pulse,
        "first_close_0p3mm_ms": motion,
        "final_target_at_most_0p1mm": (
            {
                "duration_to_last_command_s": round((end - low[0][0]) / 1e9, 3),
                "command_count": len(low),
                "min_median_max_width_mm": [
                    v / 1000
                    for v in (
                        min(d[0] for _, d in low_fb),
                        median(d[0] for _, d in low_fb),
                        max(d[0] for _, d in low_fb),
                    )
                ],
                "feedback_effort": effort(low_fb),
            }
            if low_fb
            else None
        ),
        "arm_status_histogram": dict(arm),
        "joint_position_spans_deg": spans,
        "error_frames_in_window": sum(bool(cid & 0x20000000) for _, cid, _, _ in active),
        "stop_or_enable_frames_in_window": sum(cid in (0x150, 0x471) for _, cid, _, _ in active),
    }


def sdk_check(export, official):
    files = export["files"]
    for name, record in files.items():
        assert record["source"].splitlines() == (official / name).read_text(encoding="utf-8").splitlines()
        assert sha256(record["source"].encode("utf-8")).hexdigest() == record["sha256"]
    source = files["interface/piper_interface_v3.py"]["source"]
    methods = {f.name: f for f in walk(parse(source)) if isinstance(f, FunctionDef)}
    return {
        "version": export["version"],
        "official_files_matching_installed": {k: v["sha256"] for k, v in files.items()},
        "sdk_methods": {
            k: methods[k].lineno
            for k in (
                "GetArmGripperCtrl_V3",
                "_UpdateArmGripperCtrl_V3",
                "GetArmGripperMsgs_V3",
                "_UpdateArmGripperState_V3",
            )
        },
    }


def recent_field(path):
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    sent = [r for r in rows if r.get("mode") == "execute"]
    last_small = max(i for i, r in enumerate(sent) if r["measured_at_send"][6] - r["sent"][6] < 4)
    anchor = sent[last_small + 1]
    t0 = anchor["send_finished_ns"]
    width0 = anchor["measured_at_send"][6]
    readings = {}

    def collect(fb):
        raw = fb["torque_nm"] * 1000
        assert abs(raw - round(raw)) < 1e-6
        point = (round(fb["width_mm"] * 1000), round(raw))
        assert readings.setdefault(fb["stamp_ns"], point) == point

    for r in sent:
        collect(r["gripper_at_image"])
        collect(r["gripper_control"]["feedback"])
    collect(next(r["last_check"]["feedback"] for r in rows if r["event"] == "gripper_end"))
    active = sorted((t, d) for t, d in readings.items() if t >= t0)
    start = next(r["mono_ns"] for r in rows if r["event"] == "start")
    wall_offset = rows[0]["wall_ns"] - rows[0]["mono_ns"]
    return {
        "archive_sha256": sha256(content).hexdigest(),
        "start": datetime.fromtimestamp(
            (start + wall_offset) / 1e9, timezone(timedelta(hours=8))
        ).isoformat(timespec="milliseconds"),
        "anchor_seconds_after_start": round((t0 - start) / 1e9, 3),
        "anchor_target_mm": round(anchor["sent"][6], 3),
        "initial_width_mm": width0,
        "last_width_mm": active[-1][1][0] / 1000,
        "peak_closing_feedback_counts": min(d[1] for _, d in active),
        "first_minus_200_counts_ms": next(
            (round((t - t0) / 1e6, 1) for t, d in active if d[1] <= -200), None
        ),
        "first_close_0p3mm_ms": next(
            (round((t - t0) / 1e6, 1) for t, d in active if round(width0 * 1000) - d[0] >= 300), None
        ),
        "max_feedback_gap_ms": round(max(b - a for (a, _), (b, _) in zip(active, active[1:])) / 1e6, 2),
    }


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("windows_zip", type=Path)
    cli.add_argument("sdk_export", type=Path)
    cli.add_argument("official_package_source", type=Path)
    cli.add_argument("field_logs", nargs="+", type=Path)
    args = cli.parse_args()
    with ZipFile(args.windows_zip) as src:
        ready = json.loads(src.read("info.json"))["ready"]
        assert FRAME.size == ready["record_size"] and ready["record_layout"] == FRAME.format
        names = sorted(n for n in src.namelist() if n.endswith(".bin"))
        report = {
            "subscribed_ids": [hex(i) for i in ready["ids"]],
            "capture_error_mask": ready["errors_mask"],
            "windows": [window(src.read(n), int(n[:-4])) for n in names],
        }
    report["sdk"] = sdk_check(json.loads(args.sdk_export.read_bytes()), args.official_package_source)
    report["nov25_field"] = [recent_field(path) for path in sorted(args.field_logs)]
    print(json.dumps(report, indent=2))
