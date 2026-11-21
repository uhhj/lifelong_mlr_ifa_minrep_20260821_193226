"""Cross-check an existing B1 archive against a receive-only sample of can0."""
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import select
import socket
import statistics
import struct
import subprocess
import sys
import time

work = Path(sys.argv[1])
zone = timezone(timedelta(hours=8))


def clock():
    return datetime.now(zone).isoformat(timespec="milliseconds")


def link():
    result = subprocess.run(["/usr/sbin/ip", "-j", "-d", "-s", "link", "show", "dev", "can0"],
                            capture_output=True, text=True)
    if result.returncode:
        return {"missing": True}
    dev, = json.loads(result.stdout)
    info = dev["linkinfo"]["info_data"]
    stats = dev.get("stats64", dev.get("stats", {}))
    return dict(state=info.get("state"), rate=info.get("bittiming", {}).get("bitrate"),
                berr_counter=info.get("berr_counter"), rx=stats.get("rx"), tx=stats.get("tx"))


def tail(path, size=2**20):
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        end = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, end-size))
        if stream.tell():
            stream.readline()
        lines = stream.read(end-stream.tell()).splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


def summary(values):
    if not values:
        return None
    values = sorted(values)
    return dict(median=round(statistics.median(values), 3),
                p95=round(values[math.ceil(.95*len(values))-1], 3), max=round(values[-1], 3))


def listen():
    joint_ids = (0x2a5, 0x2a6, 0x2a7)
    positions = {cid: [] for cid in (*joint_ids, 0x2a8)}
    selected = (*positions, 0x2a1, *range(0x150, 0x160), 0x470, 0x471)
    error, commands, status = Counter(), Counter(), Counter()
    overflow = 0
    timestamp_type = getattr(socket, "SO_TIMESTAMPNS", 35)
    overflow_type = getattr(socket, "SO_RXQ_OVFL", 40)
    offset = time.time_ns()-time.monotonic_ns()
    with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
        sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                        b"".join(struct.pack("=II", cid, 0xc00007ff) for cid in selected))
        sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER, struct.pack("=I", 0x1fffffff))
        sock.setsockopt(socket.SOL_SOCKET, timestamp_type, 1)
        sock.setsockopt(socket.SOL_SOCKET, overflow_type, 1)
        sock.bind(("can0",))
        begin = time.monotonic_ns()
        while time.monotonic_ns()-begin < 8_000_000_000:
            if not select.select([sock], [], [], .1)[0]:
                continue
            raw, anc, flags, _ = sock.recvmsg(16, 128)
            stamp = None
            for level, kind, data in anc:
                if level == socket.SOL_SOCKET and kind == timestamp_type:
                    sec, ns = struct.unpack_from("@ll", data)
                    stamp = sec*10**9+ns-offset
                if level == socket.SOL_SOCKET and kind == overflow_type:
                    overflow = struct.unpack_from("=I", data)[0]
            cid, dlc, data = struct.unpack("=IB3x8s", raw)
            if cid & 0x20000000:
                error[hex(cid)] += 1
            elif flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or dlc != 8 or stamp is None:
                error["truncated_or_unstamped"] += 1
            elif cid in positions:
                values = struct.unpack(">ii", data) if cid in joint_ids else struct.unpack_from(">i", data)
                positions[cid].append((stamp, values))
            elif cid == 0x2a1:
                status[data.hex()] += 1
            else:
                commands[hex(cid)] += 1
        end = time.monotonic_ns()
    return begin, end, positions, dict(error=error, commands=commands, status=status, overflow=overflow)


def audit():
    before = link()
    start = clock()
    if before.get("missing"):
        return dict(start=start, can_before=before, end=clock())
    begin, end, positions, bus = listen()
    stop = clock()
    out = dict(start=start, end=stop, can_before=before, can_after=link(), bus=bus,
               seconds=round((end-begin)/1e9, 3))
    time.sleep(.15)  # Let predictions already in flight at the window boundary reach the main log.
    path = max((work/"policy_runs").glob("*_B1.jsonl"), key=lambda p:p.stat().st_mtime_ns, default=None)
    if not path:
        return out
    root = path.with_suffix("")
    main = tail(path)
    indexed = tail(root/"inputs.jsonl", 2**21)
    index = {r["stamp"]: r for r in indexed}
    decision = [r for r in main if r.get("event") == "prediction" and begin <= r["stamp"] <= end]
    out.update(run=path.stem, last_event=main[-1]["event"] if main else None,
               end_reason=main[-1].get("detail") if main else None, decisions=len(decision),
               archived_in_window=sum(begin <= s <= end for s in index),
               last_archived_age_s=round((time.monotonic_ns()-max(index))/1e9, 3) if index else None)
    if not decision:
        return out
    flaws = Counter()
    agree = 0
    measured = 0
    ages, skews, lags, raw_diffs = [], [], [], []
    prev = {}
    gaps = Counter()
    times = {k:[t for t,_ in rows] for k,rows in positions.items()}
    matched = []
    for r in decision:
        stamp = r["stamp"]
        s = index.get(stamp)
        if not s:
            flaws["missing_index"] += 1
            continue
        matched.append(s)
        if not (r["full"] and r["mode"] == s["mode_at_query"] == "preview" and "sent" not in r):
            flaws["mode_or_history_flag"] += 1
        if not all(r[k] == s[k] for k in ("q","p","feedback_ns","images")):
            flaws["row_disagreement"] += 1
        if not all(len(s[k]) == 7 and all(map(math.isfinite,s[k])) for k in ("q","p")):
            flaws["invalid_vector"] += 1
        if not (s["history"] == sorted(set(s["history"])) and len(s["history"]) == 8
                and s["history"][-1] == stamp and all(h in index for h in s["history"])):
            flaws["history"] += 1
        cam = s["cameras"]
        arrivals = [cam[k]["host_monotonic_ns"] for k in ("external","wrist")]
        skew = max(arrivals)-min(arrivals)
        if not (skew == s["skew_ns"] <= 20_000_000 and max(arrivals) == stamp
                and all(v["read_started_monotonic_ns"] <= v["host_monotonic_ns"] for v in cam.values())):
            flaws["camera_timing"] += 1
        for k,v in cam.items():
            seq = v["sequence"]
            if k in prev:
                if seq <= prev[k]: flaws["camera_sequence"] += 1
                gaps[k] += max(0, seq-prev[k]-1)
            prev[k] = seq
        for k in ("scene","model"):
            f = root/s["images"][k]
            if not f.is_file() or not f.stat().st_size: flaws["missing_"+k] += 1
        ft = s["feedback_ns"]
        age = stamp-min(ft.values())
        if not (0 <= age <= 60_000_000 and max(ft.values()) <= stamp
                and ft["joint_first_ns"] <= ft["joint_last_ns"] and stamp <= s["predicted_ns"] <= r["mono_ns"]):
            flaws["feedback_or_prediction_timing"] += 1
        skews.append(skew/1e6)
        ages.append(age/1e6)
        lags.append((r["mono_ns"]-stamp)/1e6)
        # Independently captured kernel timestamps identify the joint triple and gripper frame.
        # The center joint frame must lie within the same triple, not just before the image.
        if min(ft.values()) < begin+1_000_000: continue
        readings=[]
        diffs=[]
        for cid,name in ((0x2a5,"joint_first_ns"),(0x2a6,"joint_last_ns"),
                         (0x2a7,"joint_last_ns"),(0x2a8,"gripper_ns")):
            t=ft[name]
            ix=bisect_right(times[cid],t)
            options=positions[cid][max(0,ix-2):ix+2]
            if cid==0x2a6:
                options=[x for x in options if ft["joint_first_ns"]-100_000<=x[0]<=t+100_000]
            if not options: break
            got, vals=min(options,key=lambda x:abs(x[0]-t))
            if cid!=0x2a6: diffs.append(abs(got-t)/1e6)
            readings.extend(vals)
        measured += 1
        if len(readings)==7 and max(diffs)<.5:
            compare=[math.radians(v/1000) for v in readings[:6]]+[readings[6]/1000]
            if all(abs(x-y)<1e-8 for x,y in zip(compare,s["q"])): agree+=1
            raw_diffs.extend(diffs)
    out.update(matched=len(matched), flaws=flaws, camera_skew_ms=summary(skews),
               oldest_feedback_age_ms=summary(ages), decision_latency_ms=summary(lags),
               skipped_camera_seq=gaps, independent_checked=measured, independent_agree=agree,
               independent_stamp_diff_ms=summary(raw_diffs))
    if matched:
        row=matched[-1]
        out["example"]={k:row[k] for k in ("stamp","q","p","images")}
    return out


if __name__=="__main__":
    print(json.dumps(audit(),ensure_ascii=False,allow_nan=False))
