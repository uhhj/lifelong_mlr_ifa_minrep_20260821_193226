"""Read only archived B1 logs, calibration and expert labels. No device APIs."""
import collections
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np


RUNS = ("212732_689056", "213025_690417", "213755_694001")


def lines(path):
    with path.open(encoding="utf-8") as src:
        for line in src:
            yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mm(value):
    return round(float(value), 2)


def percentiles(values):
    return [mm(v) for v in np.percentile(values, [0, 10, 50, 90, 100])]


def audit(work):
    diag = work / "offline_diagnostics"
    sys.path.insert(0, str(diag / "b1_spatial_ablation_20261121_2034"))
    from fit_gripper import flange
    reference = diag / "b1_vision_contrasts_20261124"
    geo = json.loads((reference / "final.json").read_bytes())["geometry"]

    def xyz(q):
        poses = flange(np.rad2deg(np.asarray(q, dtype=float)[..., :6]))
        return poses[..., :3, 3] + poses[..., :3, :3] @ geo["tip_mm"]

    manifest = diag / "b1_ode_lerobot_20261123/selection.json"
    selection = json.loads(manifest.read_bytes())
    labels = json.loads((reference / "points.json").read_bytes())
    for i, correction in json.loads((reference / "annotations.json").read_bytes()).items():
        labels[int(i)].update(correction)
    ranking = {r["id"]: r for r in selection["ranking"]}
    expert_z = []
    expert_command_z = []
    for lab in labels:
        if lab["episode"] not in selection["selected_ids"] or lab["contact"] is None or lab["close"] is None:
            continue
        ref = ranking[lab["episode"]]
        source = work / "batches/B1_20261123" / ref["source"] / "alignment.jsonl"
        assert sha(source) == ref["alignment_sha256"]
        rows = list(lines(source))[ref["first"]:]
        expert_z.append(xyz(rows[lab["contact"]]["feedback"]["joints_rad"])[2])
        expert_command_z.append(xyz(rows[lab["close"]]["feedback"]["joints_rad"])[2])
    result = {"expert_selected": len(selection["selected_ids"]), "expert_comparable": len(expert_z),
              "expert_z_at_first_sustained_width_le49_mm_p0_p10_p50_p90_p100": percentiles(expert_z),
              "expert_z_at_first_close_target_le55_mm_p0_p10_p50_p90_p100": percentiles(expert_command_z),
              "selection_sha256": sha(manifest), "runs": []}
    root = work / "policy_runs"
    prior = next(root.glob("20261127_202905_658496_B1-*.jsonl"))
    prior_sent = {r["stamp"] for r in lines(prior) if "sent" in r and r.get("mode") == "execute"}
    prior_offsets = collections.Counter(r["plan"]["offset"] for r in lines(root / prior.stem / "inputs.jsonl")
                                        if r["stamp"] in prior_sent)
    assert sum(prior_offsets.values()) == len(prior_sent)
    result["before_gate_plan_offsets"] = dict(prior_offsets)
    for key in RUNS:
        path = next(root.glob("20261127_" + key + "_B1-*.jsonl"))
        bundle = list(lines(path))
        head, end = bundle[0], bundle[-1]
        assert head["gripper_control"]["mode"] == "pika_fixed_grasp_place" and "6.3" in end["detail"]
        rows = [r for r in bundle if r.get("mode") == "execute" and "sent" in r]
        stamps = {r["stamp"] for r in rows}
        inputs = root / path.stem / "inputs.jsonl"
        frames = {r["stamp"]: r for r in lines(inputs) if r["stamp"] in stamps}
        assert len(frames) == len(rows)
        plans = [frames[r["stamp"]]["plan"] for r in rows]
        counts = collections.Counter(p["offset"] for p in plans)
        only_width = 0
        for j, (prev, plan) in enumerate(zip(rows, plans[1:])):
            diff = np.abs(np.array(plan["previous_sent"], dtype=np.float32) - np.array(prev["p"], dtype=np.float32))
            gap = rows[j+1]["stamp"] - prev["stamp"]
            if plan["offset"] == 0 and gap <= 50_000_000 and diff[6] > .001 and np.all(diff[:6] <= np.deg2rad(.001)):
                only_width += 1

        c = next(i for i, r in enumerate(rows) if r["pika_target_mm"] == 35)
        carrying = next(i for i, r in enumerate(rows) if r["gripper_phase"] == "carrying")
        q = np.array([r["measured_at_send"] for r in rows])
        narrow = next(i for i in range(c, len(rows)-2) if max(q[i:i+3, 6]) <= 49)
        raw, sent, measured = [xyz([r[k] for r in rows]) for k in ("p", "sent", "measured_at_send")]
        camera_tip = np.array([r["tip_xyz_mm"] for r in rows])
        assert np.max(np.abs(camera_tip - xyz([r["q"] for r in rows]))) < .001
        alignment = [p["alignment"] for p in plans]
        last_guide = [x for x in alignment[:c+1] if x["applied"]]
        tick = rows[0]["gate_execution"]
        wall = head["wall_ns"] + tick - head["mono_ns"]
        result["runs"].append({
            "session": path.stem, "policy_log_sha256": sha(path), "inputs_sha256": sha(inputs),
            "execution_cst": datetime.fromtimestamp(wall/1e9, timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
            "timed_end_feedback": end["detail"], "dispatch_count": len(rows), "plan_offsets": dict(counts),
            "width_only_replans_under_50ms": only_width, "adjacent_execution_pairs": len(rows)-1,
            "start_width_mm": mm(q[0, 6]), "close_seconds": round((rows[c]["stamp"]-tick)/1e9, 3),
            "close_command_feedback_z_mm": mm(measured[c, 2]), "sustained_width_le49_z_mm": mm(measured[narrow, 2]),
            "through_carrying_entry_min_z_mm": {k: mm(min(v[:carrying+1, 2])) for k, v in
                                                (("predicted", raw), ("sent", sent), ("feedback", measured))},
            "carry_entry_seconds_after_close": round((rows[carrying]["stamp"]-rows[c]["stamp"])/1e9, 3),
            "max_z_gain_after_carry_entry_relative_to_close_mm": mm(max(camera_tip[carrying:, 2]) - camera_tip[c, 2]),
            "placement_phase_end": rows[-1]["gripper_phase"], "gripper_writes_mm": [
                r["pika_target_mm"] for r in rows if r["pika_target_mm"] is not None],
            "alignment_applied_ticks": sum(a["applied"] for a in alignment),
            "alignment_phase_counts": dict(collections.Counter(a["phase"] for a in alignment)),
            "last_alignment_distance_mm": mm(last_guide[-1]["distance_mm"]) if last_guide else None,
            "camera_timestamp_skew_p95_ms": mm(np.percentile([f["skew_ns"]/1e6 for f in frames.values()], 95)),
        })
    return result


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, ensure_ascii=True))
