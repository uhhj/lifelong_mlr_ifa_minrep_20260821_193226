"""Check B1 mode, joint tracking, gripper response and coverage of the passive listener."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import sys

SESSIONS = ("20261125_142201_3055878_B1-ode50-aligned-20261125", "20261125_142343_3056561_B1-ode50-aligned-20261125")
TZ = timezone(timedelta(hours=8))


def rows(p):
    return [json.loads(line) for line in p.read_bytes().splitlines()]


def clock(t):
    return datetime.fromtimestamp(t, TZ).isoformat(timespec="milliseconds")


def span(values):
    return [round(v,4) for v in (min(values), median(values), max(values))]


def audit(root, wire_path, replay):
    wire = [line.split() for line in wire_path.read_bytes().splitlines() if line.strip()]
    wt = [float(c[0][1:-1]) for c in wire]
    result = dict(passive_times=[clock(min(wt)),clock(max(wt))],
                  passive_frame_counts=dict(Counter(c[-1].split(b"#")[0].decode() for c in wire)), trials=[])
    for name, rerun in zip(SESSIONS, replay["runs"]):
        assert name == rerun["name"]
        lp,ip = root/(name+".jsonl"),root/name/"inputs.jsonl"
        events = rows(lp)
        inputs = [r for r in rows(ip) if r["mode_at_query"]=="execute"]
        by = {r["stamp"]:r for r in inputs}
        sent = [r for r in events if r.get("mode")=="execute"]
        h = events[0]
        begin = next(r for r in events if r["event"]=="start")
        end = events[-1]
        assert "夹爪响应等待超时" in end["detail"] and "已收到从臂 STOP 反馈" in end["detail"]
        def seconds(ns):
            return (ns-begin["mono_ns"])/1e9
        def wall(ns):
            return (ns + h["wall_ns"] - h["mono_ns"])/1e9
        previous = begin["initial"]
        gaps = []
        for r in sent:
            x = by[r["stamp"]]
            assert x["p"] == r["p"] and x["q"] == r["q"] and x["images"]==r["images"]
            assert r["sent_ids"] == [0x155,0x156,0x157,0x159]
            assert r["gripper_control"]["previous_sent_mm"] == previous[6]
            gaps.append(max(abs(math.degrees(m-v)) for m,v in zip(r["measured_at_send"][:6],previous[:6])))
            previous = r["sent"]
        waiting = [r for r in sent if r["gripper_control"]["phase"]=="waiting"]
        w = [r["measured_at_send"][6] for r in waiting]
        force = [r["gripper_control"]["feedback"]["torque_nm"] for r in waiting]
        fw = [r for r in inputs if r["plan"]["alignment"]["applied"]]
        times = [seconds(r["stamp"]) for r in fw]
        distances = [r["plan"]["alignment"]["distance_mm"] for r in fw]
        plans = rerun["plans"]
        wrong = [p for p in plans if p["endpoint_change_in_raw_distance_mm"]>0]
        closed = next((x for x in inputs if x["plan"]["alignment"]["phase"]=="closing_chunk"), None)
        item = dict(start=clock(wall(begin["mono_ns"])),end=clock(wall(end["mono_ns"])), session=name,
                    model_sha=h["policy"]["sha256"], recordings=[sha256(p.read_bytes()).hexdigest() for p in (lp,ip)],
                    passive_frames_during_run=sum(wall(begin["mono_ns"])<=t<=wall(end["mono_ns"]) for t in wt),
                    accepted_grip_start_range=h["policy"]["start_range"][6],close_mm=h["policy"]["alignment"]["close_mm"],
                    start_width=begin["initial"][6],peak_width_mm=max(r["measured_at_send"][6] for r in sent),
                    execution_inputs=len(inputs),sent=len(sent),phase=dict(Counter(r["plan"]["alignment"]["phase"] for r in inputs)),
                    first_closing_chunk_s=seconds(closed["stamp"]) if closed else None,
                    last_applied_s=times[-1] if times else None,
                    raw_distance_first_min_last_mm=[distances[0],min(distances),distances[-1]] if fw else None,
                    block_uv_range=[[min(r["plan"]["alignment"]["block_uv"][j] for r in fw),
                                     max(r["plan"]["alignment"]["block_uv"][j] for r in fw)] for j in range(2)] if fw else None,
                    plans=len(plans),away_endpoint_plans=len(wrong),
                    away_plan_interval_s=[wrong[0]["seconds"],wrong[-1]["seconds"]] if wrong else None,
                    max_away_mm=max((p["endpoint_change_in_raw_distance_mm"] for p in wrong),default=None),
                    max_replay_error_mm=max((p["max_reconstruction_error_mm"] for p in plans),default=None),
                    wait_start_s=seconds(waiting[0]["mono_ns"]),wait_commands=len(waiting),
                    wait_target_mm=span([r["sent"][6] for r in waiting]),wait_width_mm=span(w),
                    wait_torque_nm=span(force),wait_drive_states=sorted({r["gripper_control"]["feedback"]["status"] for r in waiting}),
                    max_joint_gap_deg=max(gaps),max_camera_skew_ms=max(r["skew_ns"] for r in inputs)/1e6,
                    max_send_feedback_age_ms=max(r["mono_ns"]-r["measured_at_send_ns"]["gripper_ns"] for r in sent)/1e6,
                    outcome=end["detail"])
        result["trials"].append(item)
    return result


if __name__=="__main__":
    print(json.dumps(audit(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_bytes())),
                     ensure_ascii=False,indent=2))
