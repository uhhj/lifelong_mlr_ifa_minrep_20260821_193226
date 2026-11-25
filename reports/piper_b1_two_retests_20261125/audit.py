"""Audit the two completed B1 runs using the paired execution log and model input index."""
from collections import Counter
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import sys

SESSIONS = ("20261125_132907_3026012_B1-ode50-aligned-20261125",
            "20261125_133047_3026715_B1-ode50-aligned-20261125")


def records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def spread(values):
    return [round(min(values), 4), round(median(values), 4), round(max(values), 4)]


def audit(root, name):
    log = root / (name + ".jsonl")
    index = root / name / "inputs.jsonl"
    events = records(log)
    inputs = [r for r in records(index) if r["mode_at_query"] == "execute"]
    start = next(r for r in events if r["event"] == "start")
    rows = [r for r in events if r.get("mode") == "execute"]
    end = events[-1]
    header = events[0]
    by_stamp = {r["stamp"]:r for r in inputs}
    assert len(inputs) == len(by_stamp) == len(rows)
    assert all(r["sent_ids"] == [0x155, 0x156, 0x157, 0x159] for r in rows)
    assert end["event"] == "end" and "夹爪响应等待超时" in end["detail"] and "已收到从臂 STOP 反馈" in end["detail"]
    previous = start["initial"]
    joint_gaps, waiting_requests = [], []
    for r in rows:
        x = by_stamp[r["stamp"]]
        assert r["p"] == x["p"] and r["q"] == x["q"] and r["images"] == x["images"]
        ctrl = r["gripper_control"]
        fb = ctrl["feedback"]
        assert (fb["width_mm"] == r["measured_at_send"][6] and fb["stamp_ns"] == r["measured_at_send_ns"]["gripper_ns"]
                and r["gripper_at_image"]["stamp_ns"] == x["feedback_ns"]["gripper_ns"] and ctrl["previous_sent_mm"] == previous[6])
        joint_gaps.append(max(abs(math.degrees(p-m)) for p,m in zip(previous[:6], r["measured_at_send"][:6])))
        previous = r["sent"]

    def seconds(ns):
        return round((ns-start["mono_ns"])/1e9, 6)

    def clock(ns):
        wall = (header["wall_ns"] + ns - header["mono_ns"]) / 1e9
        return datetime.fromtimestamp(wall, timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")

    wait = [r for r in rows if r["gripper_control"]["phase"] == "waiting"]
    assert rows[-len(wait):] == wait and len({tuple(r["sent"]) for r in wait}) == 1
    assert not any(r["gripper_control"]["phase"] == "holding" for r in rows)
    held = wait[0]["sent"]
    assert rows[rows.index(wait[0])-1]["sent"] == held
    for r in wait:
        fb = r["gripper_control"]["feedback"]
        requested = min(held[6], max(r["p"][6], fb["width_mm"]-6))
        waiting_requests.append(requested)
    assert all(value == held[6] for value in waiting_requests)
    fbs = [r["gripper_control"]["feedback"] for r in wait]
    times = sorted({f["stamp_ns"] for f in fbs})
    on_image = [r["gripper_at_image"] for r in rows if wait[0]["mono_ns"] <= r["gripper_at_image"]["stamp_ns"]]
    wait_observed = {f["stamp_ns"]:f for f in fbs + on_image}
    contact = [f for f in wait_observed.values() if 35 <= f["width_mm"] <= 44 and -1.20 <= f["torque_nm"] <= -.80]
    corrected = [r for r in inputs if r["plan"]["alignment"]["applied"]]
    phase_change = next(r for r in rows if r["grip_plan_phase"] == "closing_chunk")
    sample_target = next(r for r in rows if r["sent"][6] <= 55)
    actual = [r["gripper_control"]["feedback"] for r in rows]
    stability_start = len(actual)-1
    while stability_start and abs(actual[stability_start-1]["width_mm"] - actual[-1]["width_mm"]) <= .11:
        stability_start -= 1
    delay = [(r["mono_ns"]-r["stamp"])/1e6 for r in rows]
    age = [(r["mono_ns"]-f["stamp_ns"])/1e6 for r,f in zip(rows,actual)]
    return dict(session=name, model_sha256=header["policy"]["sha256"], guard=header["gripper_control"],
                started=clock(start["mono_ns"]), stopped=clock(end["mono_ns"]), runtime_s=seconds(end["mono_ns"]),
                outcome=end["detail"], sdk_submissions=len(rows), synchronized_inputs=len(inputs),
                plan_counts=dict(Counter(r["grip_plan_phase"] for r in rows)),
                estimated_distance_first_last_mm=[r["plan"]["alignment"]["distance_mm"] for r in (corrected[0],corrected[-1])],
                closing_plan_s=seconds(phase_change["mono_ns"]), first_sent_55_s=seconds(sample_target["mono_ns"]),
                width_start_mm=start["initial"][6], final_width_within_0p11mm_since_s=seconds(actual[stability_start]["stamp_ns"]),
                wait_start_s=seconds(wait[0]["mono_ns"]), wait_deadline_s=seconds(wait[0]["gripper_control"]["deadline_ns"]),
                wait_commands=len(wait), wait_all_seven_targets_constant=True, held_gripper_mm=held[6],
                wait_measured_mm=spread([f["width_mm"] for f in fbs]), wait_effort_nm=spread([f["torque_nm"] for f in fbs]),
                wait_raw_policy_mm=spread([r["p"][6] for r in wait]), unique_send_feedback_times=len(times),
                observed_wait_feedback_times=len(wait_observed), qualifying_contact_samples=len(contact),
                max_gap_between_send_feedback_ms=max(b-a for a,b in zip(times,times[1:]))/1e6,
                fresh_close_request_below_mm=held[6]+6, sdk_feedback_status=sorted({f["status"] for f in actual}),
                whole_execution_sampled_effort_nm=spread([f["torque_nm"] for f in actual]),
                max_joint_gap_deg=max(joint_gaps), max_camera_skew_ms=max(x["skew_ns"] for x in inputs)/1e6,
                input_to_check_ms=spread(delay), send_feedback_age_ms=spread(age),
                run_sha256=sha256(log.read_bytes()).hexdigest(), inputs_sha256=sha256(index.read_bytes()).hexdigest())


if __name__ == "__main__":
    print(json.dumps([audit(Path(sys.argv[1]),n) for n in SESSIONS], ensure_ascii=False, indent=2))
