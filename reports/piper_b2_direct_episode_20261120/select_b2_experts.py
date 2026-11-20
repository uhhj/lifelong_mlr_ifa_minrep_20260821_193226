"""Rank the 52 audited B2 episodes and build their 50-episode training manifest."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from decimal import Decimal

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"piper_b1_direct_episode_20261120"))
from b1_selection_metrics import closed_cycles, motion_pauses, training_motion
from select_b1_experts import percentiles


def candidate_pool(audit):
    pool = [r for r in audit["episodes"] if r["assessment"] == "include"]
    if audit["task"] != "B2" or len(pool) != 52 or len({r["episode_id"] for r in pool}) != 52 or len({r["alignment_sha256"] for r in pool}) != 52:
        raise ValueError("Expected 52 distinct approved B2 episodes")
    return pool


def measure(work, item, release_row):
    name = item["episode_id"]
    episode = work/"recordings"/name
    meta = json.loads((episode/"episode_summary.json").read_text())
    raw = (episode/"alignment.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["alignment_sha256"]:
        raise ValueError("Alignment changed: "+name)
    if (meta["task"],meta["outcome"],meta["action_source"]) != ("B2","success","direct_can_rx"):
        raise ValueError("Episode summary changed: "+name)
    rows = [json.loads(line) for line in raw.splitlines()]
    start,end = item["training_range_inclusive"]
    if not (0 <= start <= release_row <= end == len(rows)-1 and meta["pairs"] == len(rows) == item["pairs"]
            and meta["valid_targets"] == item["valid_actions"] == end-start+1
            and all(r["feedback"] and (r["target"] is not None) == (i>=start) for i,r in enumerate(rows))):
        raise ValueError("Invalid training interval: "+name)
    ns=np.array([r["sample_monotonic_ns"] for r in rows],dtype="i8")
    folders=("camera/color/overhead","camera/color/wrist","array/float32/follower_target","array/float32/follower_feedback")
    keys=None
    for folder in folders:
        paths=[f for f in (episode/"episode0"/folder).iterdir() if f.stem.replace(".","",1).isdigit()]
        names=sorted((f.stem for f in paths),key=Decimal)
        if len(names)!=len(rows) or (keys is not None and keys!=names):
            raise ValueError("Mismatched file indices: "+name)
        if any(not f.is_file() or f.stat().st_size==0 for f in paths):
            raise ValueError("Missing or empty file: "+name)
        if keys is None and any(abs(int(Decimal(k)*10**9)-int(t))>500 for k,t in zip(names,ns)):
            raise ValueError("File index outside timestamp tolerance: "+name)
        keys=names
    q=np.rad2deg([r["feedback"]["joints_rad"] for r in rows])
    command=np.rad2deg([r["target"]["joints_rad"] for r in rows[start:release_row+1]])
    rms=np.sqrt(np.mean((command-q[start:release_row+1])**2,axis=1))
    widths=np.array([r["feedback"]["gripper_mm"] for r in rows])
    smooth=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(widths,2,mode="edge"),5),axis=1)
    low=float(smooth[start:release_row+1].min())
    secs=(ns-ns[0])/1e9
    skips=[]
    for role in ("overhead","wrist"):
        diffs=np.diff([r["cameras"][role]["sequence"] for r in rows[start:]])
        if np.any(diffs<=0): raise ValueError("Nonincreasing camera sequence: "+name)
        skips.append(int(np.maximum(diffs-1,0).sum()))
    skew=[abs(r["cameras"]["wrist"]["monotonic_ns"]-r["cameras"]["overhead"]["monotonic_ns"])/1e6 for r in rows[start:]]
    return dict(episode_id=name,alignment_sha256=item["alignment_sha256"],release_evidence_row=release_row,
                grip_cycles=closed_cycles(smooth,start,release_row,low+3,low+15),
                pause_excess_s=motion_pauses(q,smooth,secs,start,release_row),skips_in_valid=skips,
                max_gap_ms=round(float(np.diff(ns[start:]).max())/1e6,3),
                p95_skew_ms=round(float(np.percentile(skew,95)),3),p95_tracking_deg=round(float(np.percentile(rms,95)),3),
                **training_motion(rows,start,release_row))


def select(audit, measurements):
    pool=candidate_pool(audit)
    if set(measurements)!={i["episode_id"] for i in pool}: raise ValueError("Incomplete measurements")
    ranked=[]
    for i in pool:
        m=measurements[i["episode_id"]]
        if m["alignment_sha256"]!=i["alignment_sha256"] or m["episode_id"]!=i["episode_id"]: raise ValueError("Unbound measurement")
        issues=[]
        if m["grip_cycles"]>1: issues.append("repeated_grip_cycle")
        if m["pause_excess_s"]>=.5: issues.append("mid_task_pause")
        if m["settled_tail_s"]>=2 and m["settled_tail_fraction"]>=.20: issues.append("long_idle_tail")
        if m["pre_release_closing_mm"]<5: issues.append("limited_closing")
        if m["max_gap_ms"]>=60: issues.append("long_camera_gap")
        ranked.append(dict(**m,first_action_row=i["leading_null_actions"],recorded_pairs=i["pairs"],quality_flags=issues))
    keys=("max_gap_ms","p95_skew_ms","p95_tracking_deg")
    tables={k:percentiles(ranked,k) for k in keys}
    for r in ranked:
        r["timing_tracking_percentile"]=round(sum(tables[k][r[k]] for k in keys)/3,8)
    ranked.sort(key=lambda r:(len(r["quality_flags"]),sum(r["skips_in_valid"]),r["timing_tracking_percentile"],r["max_gap_ms"],r["episode_id"]))
    entries=[]
    for rank,r in enumerate(ranked,1):
        r["rank"]=rank
        if rank<=50:
            name=r["episode_id"]; n=r["recorded_pairs"]; start=r["first_action_row"]
            entries.append(dict(rank=rank,episode_id=name,episode_dir="episodes/"+name,alignment_sha256=r["alignment_sha256"],
                                recorded_pairs=n,first_action_row=start,last_action_row=n-1,action_count=n-start))
    stamp=datetime.fromisoformat(audit["inventory_at"]).strftime("%Y%m%d_%H%M%S")
    manifest=dict(task="B2",instruction="夹起木块，放进右侧浅色碟",set_name="B2_top50_"+stamp,
                  inventory_at=audit["inventory_at"],indexing="zero_based_inclusive_alignment_jsonl",episode_count=len(entries),
                  total_action_count=sum(r["action_count"] for r in entries),episodes=entries)
    return manifest,dict(inventory_at=audit["inventory_at"],candidate_count=len(ranked),candidates=ranked)


if __name__=="__main__":
    audit, measurements = [json.loads(Path(p).read_text(encoding="utf8")) for p in sys.argv[1:3]]
    manifest,ledger=select(audit,measurements)
    for name,obj in (("B2_expert_50.json",manifest),("B2_candidate_ranking.json",ledger)):
        (Path(sys.argv[3])/name).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode())
