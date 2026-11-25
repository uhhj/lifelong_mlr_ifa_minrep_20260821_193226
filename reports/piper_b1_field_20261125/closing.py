"""Compare the onset of B1 closing with the selected demonstrations, using archived files."""
import json
from pathlib import Path
import sys

import numpy as np


def records(path):
    return [json.loads(row) for row in path.read_bytes().splitlines()]


def spread(values):
    return dict(min=min(values), median=float(np.median(values)), max=max(values))


def compare(work):
    diagnostics = work / "offline_diagnostics"
    calibration = diagnostics / "b1_spatial_ablation_20261121_2034"
    sys.path.insert(0, str(calibration))
    from fit_gripper import flange

    tip = json.loads((calibration / "gripper_calibration.json").read_bytes())["tcp_flange_mm"]

    def height(joint_radians):
        pose = flange(np.rad2deg(joint_radians[:6]))
        return float(pose[2, 3] + pose[2, :3] @ tip)

    session = "20261125_125325_3006646_B1-ode50-aligned-20261125"
    run = records(work / "policy_runs" / (session + ".jsonl"))
    first = next(row for row in run if "sent" in row and row["sent"][6] <= 55)
    fault = next(row for row in run if row["event"] == "target_guard_fault")
    elapsed = (fault["mono_ns"] - first["send_started_ns"]) / 1e9
    actual = first["measured_at_send"]
    selection = json.loads((diagnostics / "b1_ode_lerobot_20261123/selection.json").read_bytes())
    chosen = set(selection["selected_ids"])
    heights, j2s, durations, first_second_minima = [], [], [], []
    for entry in selection["ranking"]:
        if entry["id"] not in chosen:
            continue
        source = work / "batches/B1_20261123" / entry["source"] / "alignment.jsonl"
        rows = records(source)[entry["first"]:]
        if rows[0]["target"]["gripper_mm"] <= 55:
            continue
        onset = next((i for i, row in enumerate(rows) if row["target"]["gripper_mm"] <= 55), None)
        if onset is None:
            continue
        after = rows[onset:]
        joints = after[0]["feedback"]["joints_rad"]
        heights.append(height(joints))
        j2s.append(float(np.rad2deg(joints[1])))
        stamp = after[0]["sample_monotonic_ns"]
        first_second_minima.append(min(row["feedback"]["gripper_mm"] for row in after
                                       if row["sample_monotonic_ns"] - stamp <= 1_000_000_000))
        crossed = next((row for row in after if row["feedback"]["gripper_mm"] <= 55), None)
        if crossed is not None:
            durations.append((crossed["sample_monotonic_ns"] - stamp) / 1e9)
    return dict(
        trigger_target_width_mm=55, selected_episodes=len(chosen),
        field_tcp_z_mm=height(actual), field_j2_deg=float(np.rad2deg(actual[1])),
        field_seconds_from_first_sent_to_fault=elapsed, comparable_experts=len(heights),
        expert_tcp_z_mm=spread(heights), expert_j2_deg=spread(j2s),
        experts_with_feedback_crossing=len(durations), expert_feedback_crossing_s=spread(durations),
        experts_later_than_field_stop=sum(value > elapsed for value in durations),
        expert_min_feedback_in_first_second_mm=spread(first_second_minima),
    )


if __name__ == "__main__":
    print(json.dumps(compare(Path(sys.argv[1])), indent=2, allow_nan=False))
