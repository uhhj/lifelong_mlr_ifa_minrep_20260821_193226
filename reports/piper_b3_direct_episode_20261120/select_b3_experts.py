"""Rank the audited B3 push demonstrations using their valid training intervals."""
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def candidates(audit):
    pool = [r for r in audit["episodes"] if r["assessment"] == "candidate"]
    if (audit["task"] != "B3" or len(pool) != 51 or len({r["episode_id"] for r in pool}) != 51
            or len({r["alignment_sha256"] for r in pool}) != 51
            or any(r["field_outcome"] != "success" for r in pool)):
        raise ValueError("Expected 51 distinct audited B3 successes")
    return pool


def measure(work, item):
    name = item["episode_id"]
    ep = work / "recordings" / name
    meta = json.loads((ep / "episode_summary.json").read_text())
    raw = (ep / "alignment.jsonl").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (digest != item["alignment_sha256"] or (meta["task"], meta["outcome"], meta["action_source"])
            != ("B3", "success", "direct_can_rx")):
        raise ValueError("Source does not match frozen audit for " + name)
    rows = [json.loads(line) for line in raw.splitlines()]
    start, end = item["valid_action_range_inclusive"]
    if not (0 <= start < end == len(rows)-1 and meta["pairs"] == item["pairs"] == len(rows)
            and meta["valid_feedback"] == item["valid_feedback"] == len(rows)
            and meta["valid_targets"] == item["valid_actions"] == end-start+1
            and all(r["feedback"] and (r["target"] is not None) == (i >= start) for i,r in enumerate(rows))):
        raise ValueError("Invalid training rows for " + name)
    ns = np.array([r["sample_monotonic_ns"] for r in rows], dtype="i8")
    if np.any(np.diff(ns) <= 0):
        raise ValueError("Sample clock regression for " + name)
    keys = None
    for folder in ("camera/color/overhead", "camera/color/wrist",
                   "array/float32/follower_target", "array/float32/follower_feedback"):
        paths = [p for p in (ep / "episode0" / folder).iterdir() if p.stem.replace(".", "", 1).isdigit()]
        stamps = sorted((p.stem for p in paths), key=Decimal)
        if (len(stamps) != len(rows) or (keys is not None and keys != stamps)
                or any(not p.is_file() or p.stat().st_size == 0 for p in paths)):
            raise ValueError("Missing or mismatched files in " + name + "/" + folder)
        if keys is None and any(abs(int(Decimal(k)*10**9)-int(t)) > 500 for k,t in zip(stamps,ns)):
            raise ValueError("File timestamps changed for " + name)
        keys = stamps
    valid = rows[start:end+1]
    skips = {}
    for role in ("overhead", "wrist"):
        delta = np.diff([r["cameras"][role]["sequence"] for r in valid])
        if np.any(delta <= 0):
            raise ValueError("Non-increasing camera sequence in " + name)
        skips[role] = int((delta-1).sum())
    skew = [abs(r["cameras"]["overhead"]["monotonic_ns"] - r["cameras"]["wrist"]["monotonic_ns"])/1e6
            for r in valid]
    targets = np.array([r["target"]["joints_rad"] for r in valid])
    positions = np.array([r["feedback"]["joints_rad"] for r in valid])
    if targets.shape != (len(valid),6) or positions.shape != targets.shape:
        raise ValueError("Unexpected joint dimensions in " + name)
    rms = np.sqrt(np.mean(np.rad2deg(targets-positions)**2, axis=1))
    if not np.isfinite(rms).all():
        raise ValueError("Nonfinite joint data in " + name)
    gaps = np.diff(ns[start:end+1])/1e6
    return dict(episode_id=name, alignment_sha256=digest, first_action_row=start, last_action_row=end,
                recorded_pairs=len(rows), action_count=len(valid), skipped=skips,
                max_gap_ms=round(float(gaps.max()),3), gaps_at_least_60ms=int(np.sum(gaps>=60)),
                p95_camera_skew_ms=round(float(np.percentile(skew,95)),3),
                p95_joint_tracking_deg=round(float(np.percentile(rms,95)),3))


def select(audit, measurements):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "piper_b1_direct_episode_20261120"))
    from select_b1_experts import percentiles

    pool = candidates(audit)
    if set(measurements) != {r["episode_id"] for r in pool}:
        raise ValueError("Candidate coverage differs from the audit")
    ranking = []
    for item in pool:
        name = item["episode_id"]
        row = dict(measurements[name])
        if (row["episode_id"] != name or row["alignment_sha256"] != item["alignment_sha256"]
                or [row["first_action_row"],row["last_action_row"]] != item["valid_action_range_inclusive"]
                or row["recorded_pairs"] != item["pairs"] or row["action_count"] != item["valid_actions"]):
            raise ValueError("Measurement disagrees with the audit for " + name)
        ranking.append(row)
    metrics = ("max_gap_ms", "p95_camera_skew_ms", "p95_joint_tracking_deg")
    tables = {key: percentiles(ranking,key) for key in metrics}
    for row in ranking:
        row["mean_percentile"] = round(sum(tables[key][row[key]] for key in metrics)/len(metrics),8)
    ranking.sort(key=lambda r: (r["gaps_at_least_60ms"],sum(r["skipped"].values()),
                                r["mean_percentile"],r["max_gap_ms"],r["episode_id"]))
    entries = []
    for rank, row in enumerate(ranking,1):
        row["rank"] = rank
        if rank <= 50:
            e = {k: row[k] for k in ("rank", "episode_id", "alignment_sha256", "recorded_pairs",
                                     "first_action_row", "last_action_row", "action_count")}
            e["episode_dir"] = "episodes/" + e["episode_id"]
            entries.append(e)
    stamp = datetime.fromisoformat(audit["inventory_at"]).strftime("%Y%m%d_%H%M%S")
    manifest = dict(task="B3", instruction=audit["description"], set_name="B3_top50_"+stamp,
                    inventory_at=audit["inventory_at"], indexing="zero_based_inclusive_alignment_jsonl",
                    episode_count=len(entries), total_action_count=sum(e["action_count"] for e in entries), episodes=entries)
    return manifest, dict(inventory_at=audit["inventory_at"], candidate_count=len(ranking), candidates=ranking)


if __name__ == "__main__":
    audit, metadata = [json.loads(Path(p).read_text(encoding="utf8")) for p in sys.argv[1:3]]
    manifest, ranking = select(audit,metadata)
    for filename, document in (("B3_expert_50.json",manifest),("B3_candidate_ranking.json",ranking)):
        (Path(sys.argv[3])/filename).write_bytes((json.dumps(document,ensure_ascii=False,indent=2)+"\n").encode())
