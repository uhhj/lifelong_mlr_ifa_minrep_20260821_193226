"""Recompute selected incident metrics from archived files; no device access."""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


def jaw_distance(angle):
    theta = math.radians(180 - 43.99) - angle
    h = 0.0325 * math.sin(theta) - 0.01456
    return 1000 * (math.sqrt(0.058**2 - h**2) + 0.0325 * math.cos(theta))


def width(angle):
    return 2 * (jaw_distance(angle) - jaw_distance(0))


def check_frames(frames):
    for frame in frames:
        raw = json.loads(frame["usb_frame"])
        assert abs(width(raw["motor"]["Position"]) - frame["width_mm"]) < 1e-6
        assert raw["motor"]["Current"] == frame["current_ma"]
        assert raw["motorstatus"]["Voltage"] == frame["voltage_v"]
        assert int(raw["motorstatus"]["Status"], 16) == frame["status"]


def elapsed_ms(later, earlier):
    return round((later - earlier) / 1e6, 3)


def audit(policy, manual, control, zero, journal):
    paths = [policy, manual, control, zero]
    rows = [json.loads(line) for line in policy.read_bytes().splitlines()]
    hand, good, dropout = [json.loads(path.read_bytes()) for path in paths[1:]]
    anchor = rows[0]["wall_ns"] - rows[0]["mono_ns"]

    def beijing(stamp):
        return datetime.fromtimestamp((stamp + anchor) / 1e9,
                                      timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")

    commands = [r["pika_target_mm"] for r in rows if r.get("mode") == "execute" and "sent" in r]
    trace = next(r for r in rows if r["event"] == "pika_stop_trace")
    stop = trace["stop_requested_ns"]
    frames = trace["frames"]
    low, high = trace["bounds_mm"]
    cross = next(f for f in frames if f["width_mm"] > high)
    disabled = next(f for f in frames if f["rx_ns"] > stop and f["status"] == 0)
    assert cross["rx_ns"] < stop < disabled["rx_ns"]
    assert rows[-1]["outcome"] == "aborted"
    assert abs(commands[-1] + 12 - high) < 1e-6

    goal = hand["goals"][-1]
    req = goal["request_ns"]
    hand_frames = hand["last_motion_frames"]
    outward = next(f for f in hand_frames if f["rx_ns"] > req and f["width_mm"] > goal["start_mm"] + .1)
    assert outward["rx_ns"] < hand["disable_requested_ns"]
    for series in (frames, hand_frames, dropout["last_motion_frames"]):
        check_frames(series)
    matched = next(g for g in good["goals"] if g["target_mm"] == goal["target_mm"])
    zeros = [f for f in dropout["last_motion_frames"] if f["width_mm"] == 0]
    assert len(zeros) == 2
    z = dropout["last_motion_frames"]
    adjacent = [z[z.index(zeros[0])-1], z[z.index(zeros[-1])+1]]
    assert all(f["current_ma"] == 0 and f["status"] == 64 for f in zeros)

    pattern = re.compile(r"Gripper set to target distance ([\d.]+) mm, corresponding motor angle ([\d.]+) rad")
    matches = [(l[:19], *map(float, m.groups())) for l in journal.read_text(encoding="utf-8").splitlines()
               if (m := pattern.search(l))]
    policy_sdk = [(mm, rad) for t, mm, rad in matches if t.startswith(("2026-11-27T20:32:26", "2026-11-27T20:32:27"))]
    assert [mm for mm, _ in policy_sdk] == commands
    hand_angle, control_angle = [next(rad for t, mm, rad in matches if t == beijing(stamp)[:19] and mm == 40)
                                 for stamp in (req, matched["request_ns"])]
    assert hand_angle == control_angle
    max_roundtrip = max(abs(width(rad) - mm) for mm, rad in policy_sdk + [(40., hand_angle)])
    assert max_roundtrip < .03

    return {
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "manual": {
            "request_cst": beijing(req), "start_mm": round(goal["start_mm"], 3),
            "target_mm": goal["target_mm"], "sdk_target_rad": hand_angle,
            "first_outward_over_0_1_mm_after_request_ms": elapsed_ms(outward["rx_ns"], req),
            "outward_current_ma": outward["current_ma"],
            "outward_speed_rad_s": json.loads(outward["usb_frame"])["motor"]["Speed"],
            "disable_request_after_target_ms": elapsed_ms(hand["disable_requested_ns"], req),
            "max_feedback_mm": round(max(f["width_mm"] for f in hand_frames), 3),
            "disable_confirmed_after_request_ms": elapsed_ms(hand["disable_confirmed_ns"], hand["disable_requested_ns"]),
            "same_target_control": {"request_cst": beijing(matched["request_ns"]), "sdk_target_rad": control_angle,
                                    "settled_mm": round(matched["observed_mm"], 3)},
        },
        "policy": {
            "stop_cst": beijing(stop), "execution_writes": len(commands),
            "sdk_target_range_mm": [round(min(commands), 3), round(max(commands), 3)],
            "last_sdk_angle_rad": policy_sdk[-1][1], "active_bounds_mm": [round(low, 3), round(high, 3)],
            "first_cross_mm": round(cross["width_mm"], 3), "cross_before_stop_ms": elapsed_ms(stop, cross["rx_ns"]),
            "current_at_cross_ma": cross["current_ma"], "speed_at_cross_rad_s": json.loads(cross["usb_frame"])["motor"]["Speed"],
            "first_disabled_after_stop_ms": elapsed_ms(disabled["rx_ns"], stop),
            "max_feedback_mm": round(max(f["width_mm"] for f in frames), 3),
            "reported_end": rows[-1]["detail"],
        },
        "other_event": {"fault_cst": beijing(dropout["fault_or_stop_ns"]), "consecutive_zero_frames": len(zeros),
                        "adjacent_widths_mm": [round(f["width_mm"], 3) for f in adjacent]},
        "max_sdk_target_roundtrip_error_mm": round(max_roundtrip, 4),
        "reported_voltage_ranges_v": [[min(f["voltage_v"] for f in series), max(f["voltage_v"] for f in series)]
                                      for series in (hand_frames, frames)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "manual", "control", "zero", "journal"):
        parser.add_argument(name, type=Path)
    print(json.dumps(audit(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
