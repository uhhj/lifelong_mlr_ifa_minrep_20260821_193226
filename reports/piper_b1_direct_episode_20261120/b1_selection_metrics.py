"""Measure a saved B1 episode using the same fixed scene references for every candidate."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from check_release import files, fixed_masks, inspect


def closed_cycles(widths, start, stop, low, high):
    # A complete sequence must start open, pass the low threshold, then reopen.
    state = "waiting_open"
    completed = 0
    for width in widths[start:stop+1]:
        if state == "waiting_open" and width >= high:
            state = "open"
        elif state == "open" and width <= low:
            state = "closed"
        elif state == "closed" and width >= high:
            completed += 1
            state = "open"
    return completed


def motion_pauses(joints, widths, ts, start, stop):
    # Use a 0.20 s baseline to suppress small differences between adjacent readings.
    indices = np.arange(start, stop+1)
    baseline = np.maximum(start, np.searchsorted(ts,ts[indices]-.2))
    moving = (np.max(abs(joints[indices]-joints[baseline]),axis=1) >= .40) | (abs(widths[indices]-widths[baseline]) >= 1.0)
    active = np.flatnonzero(moving)
    if not len(active):
        return 0.0
    stationary = indices[active[0]:active[-1]+1][~moving[active[0]:active[-1]+1]]
    groups = np.split(stationary, np.flatnonzero(np.diff(stationary)>1)+1)
    return round(sum(max(0.0,ts[g[-1]]-ts[g[0]]-1.0) for g in groups if len(g)),3)


def training_motion(rows, start, release_end):
    # Detect idle padding only when both the measured and commanded joints/gripper
    # stay near their final values after a confirmed release.
    if release_end is None:
        return dict(settled_tail_s=0.0, settled_tail_fraction=0.0, pre_release_closing_mm=None)
    valid=rows[start:]
    ns=np.array([r["sample_monotonic_ns"] for r in valid],dtype="i8")
    active=np.zeros(len(valid),dtype=bool)
    for role,joint_tolerance in (("target",.25),("feedback",.30)):
        q=np.rad2deg([r[role]["joints_rad"] for r in valid])
        gripper=np.array([r[role]["gripper_mm"] for r in valid])
        active |= np.max(abs(q-q[-1]),axis=1)>joint_tolerance
        active |= abs(gripper-gripper[-1])>.8
    last_active=np.flatnonzero(active)
    first_settled=max(release_end-start,int(last_active[-1])+1 if len(last_active) else 0)
    tail=float(ns[-1]-ns[first_settled])/1e9
    width=np.array([r["feedback"]["gripper_mm"] for r in rows[start:release_end+1]])
    filtered=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(width,2,mode="edge"),5),axis=1)
    return dict(settled_tail_s=round(tail,3),settled_tail_fraction=round(tail/((ns[-1]-ns[0])/1e9),5),
                pre_release_closing_mm=round(float((np.maximum.accumulate(filtered)-filtered).max()),2))


def measure(work, name):
    cv2.setNumThreads(1)
    root = work/"recordings"
    episode = root/name
    references = [next(root.glob("20261120_"+t+"_*")) for t in ("155508","155536","155612")]
    result = inspect(episode, fixed_masks(references))
    rows = [json.loads(s) for s in (episode/"alignment.jsonl").read_text().splitlines()]
    start = next(i for i,r in enumerate(rows) if r["target"] is not None)
    time = np.asarray([r["sample_monotonic_ns"] for r in rows], dtype="i8")
    ts = (time-time[0])/1e9
    widths = np.array([r["feedback"]["gripper_mm"] for r in rows])
    joints = np.degrees(np.asarray([r["feedback"]["joints_rad"] for r in rows]))
    # Median smoothing only for threshold crossings. All timestamps and counts remain original.
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(widths,2,mode="edge"),5),axis=1)
    release = []
    for window in result["blue"]["clear_intervals"]:
        l, h = window["first"], window["last"]
        if l < start or h <= start or window["frames"] < 5:
            continue
        before = smooth[start:l+1]
        if np.median(smooth[l:h+1])-before.min() >= 5 and before.max()-before.min() >= 5:
            release.append(window)
    stop = release[0]["last"] if release else len(rows)-1
    minimum = float(smooth[start:stop+1].min())
    preopen = float(smooth[start:stop+1].max())
    skips = []
    for camera in ("overhead","wrist"):
        seq = np.array([r["cameras"][camera]["sequence"] for r in rows[start:]])
        skips.append(int(np.maximum(np.diff(seq)-1,0).sum()))
    skew = [abs(r["cameras"]["wrist"]["monotonic_ns"]-r["cameras"]["overhead"]["monotonic_ns"])/1e6 for r in rows[start:]]
    target = np.degrees(np.asarray([r["target"]["joints_rad"] for r in rows[start:stop+1]]))
    tracking_rms = np.sqrt(np.mean((target-joints[start:stop+1])**2,axis=1))
    folders = ("camera/color/overhead","camera/color/wrist","array/float32/follower_target","array/float32/follower_feedback")
    mtime = max(p.stat().st_mtime_ns for folder in folders for p in (episode/"episode0"/folder).iterdir() if p.is_file())
    return dict(valid_start=start, rows=len(rows), release_longest_frames=max((w["frames"] for w in release),default=0),
                release_first_end=stop if release else None, blue_end_found=bool(result["blue"]["stable_tail"]),
                blue_last30=result["blue"]["last30"], red_last30=result["red"]["last30"],
                red_end_found=bool(result["red"]["stable_tail"]),
                grip_close_span_mm=round(preopen-minimum,2),
                clear_grip_cycles=closed_cycles(smooth,start,stop,minimum+3,minimum+15),
                mid_pause_excess_s=motion_pauses(joints,smooth,ts,start,stop),
                skips_in_valid=skips, max_gap_ms=round(float(np.diff(ts[start:]).max()*1000),3),
                p95_skew_ms=round(float(np.percentile(skew,95)),3),
                p95_joint_tracking_deg=round(float(np.percentile(tracking_rms,95)),3),
                latest_payload_mtime_ns=mtime,**training_motion(rows,start,stop if release else None))


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1]),sys.argv[2])))
