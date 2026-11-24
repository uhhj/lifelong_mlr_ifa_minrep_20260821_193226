"""Recompute run counts, final guards, and gripper offsets from saved experiment files."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import sys


def audit(work):
    days = {f"202611{day}": {"sessions": 0, "executed": 0} for day in range(21, 25)}
    result = []
    for path in sorted((work / "policy_runs").glob("*.jsonl")):
        if path.name[:8] not in days:
            continue
        day = days[path.name[:8]]
        day["sessions"] += 1
        with path.open() as stream:
            log = [json.loads(s) for s in stream]
        start = next((r for r in log if r["event"] == "start"), None)
        rows = [r for r in log if r.get("mode") == "execute" and r["event"] == "prediction"]
        if not rows:
            continue
        assert start and log[-1]["event"] == "end"
        day["executed"] += 1
        header, end = log[0], log[-1]
        stamp = header["wall_ns"] + start["mono_ns"] - header["mono_ns"]
        bj = datetime.fromtimestamp(stamp / 1e9, timezone(timedelta(hours=8)))
        policy = header.get("policy", {})
        entry = dict(time=bj.isoformat(timespec="milliseconds"), task=header.get("task"),
                     policy=policy.get("version") or policy.get("name") or "original GOAL",
                     count=len(rows), seconds=round((end["mono_ns"]-start["mono_ns"])/1e9,3),
                     outcome=end["outcome"], detail=end["detail"])
        fault = next((r for r in log if r["event"] == "target_guard_fault"), None)
        if fault:
            entry["guard"] = [dict(axis=a["axis"], previous=round(a["previous"],3),
                                   actual=round(a["measured"],3), gap=round(a["gap"],3))
                              for a in fault["axes"]]
            if header.get("hardware_driver") and fault["axes"][0]["axis"] == "夹爪":
                prev, actual = fault["previous_sent"], fault["measured_at_send"]
                entry["max_joint_error_deg"] = round(max(abs(math.degrees(a-b)) for a,b in zip(prev[:6],actual[:6])),3)
                entry["grip_sample_age_ms"] = round((fault["mono_ns"]-fault["measured_at_send_ns"]["gripper_ns"])/1e6,3)
                entry["all_target_ids_reported"] = all(r["sent_ids"] == [341,342,343,345] for r in rows)
                first_close = next(r for r in rows if r["sent"][6] <= 55)
                entry["j2_at_first_55_deg"] = round(math.degrees(first_close["measured_at_send"][1]),3)
        result.append(entry)

    base = work / "offline_diagnostics" / "b1_ode_lerobot_20261123"
    sel = json.loads((base / "selection.json").read_bytes())
    cv = json.loads((base / "run" / "verification.json").read_bytes())
    ids = set(sel["selected_ids"])
    assert len(ids) == cv["train_episodes"]
    max_gaps, wide_counts, over_counts, pairs = [], [], [], []
    for ep in sel["ranking"]:
        if ep["id"] not in ids:
            continue
        raw = (work / "batches" / "B1_20261123" / ep["source"] / "alignment.jsonl").read_bytes()
        assert sha256(raw).hexdigest() == ep["alignment_sha256"]
        rows = [r for r in map(json.loads,raw.splitlines()) if r["target"] is not None]
        assert len(rows) == ep["frames"]
        q = [r["feedback"]["gripper_mm"] for r in rows]
        a = [r["target"]["gripper_mm"] for r in rows]
        gaps = [q[i]-a[i-1] for i in range(1,len(rows))]
        first50 = next((i for i,x in enumerate(q) if x <= 50),len(q))
        wide_counts.append(sum(q[i]>=58 and gaps[i-1]>12 for i in range(1,first50)))
        max_gaps.append(max(gaps))
        over_counts.append(sum(g>12 for g in gaps))
        pairs.append(len(gaps))
    assert len(pairs)==len(ids) and sum(pairs)+len(ids)==cv["train_rows"]
    return dict(dates=days,runs=result,latest_training=dict(
        collected_success=sel["source_successes"],trained=len(ids),rows=cv["train_rows"],
        updates=cv["global_step"],epoch=cv["epoch"],held_out=cv["validation_episodes"],
        replay_steps=cv["selected_steps"],replay_grip_mae_mm=cv["gripper_mae_mm"],
        episodes_over_12=sum(g>12 for g in max_gaps),median_peak_gap_mm=round(median(max_gaps),2),
        frames_over_12=sum(over_counts),compared_pairs=sum(pairs),
        still_open_over_12_episodes=sum(n>0 for n in wide_counts),still_open_over_12_frames=sum(wide_counts)))


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])),indent=2,ensure_ascii=False,allow_nan=False))
