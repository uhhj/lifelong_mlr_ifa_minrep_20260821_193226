"""Summarize one completed PiPER run on its recorded monotonic timeline."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import sys

SESSION = "20261125_125325_3006646_B1-ode50-aligned-20261125"


def records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def audit(directory):
    log = directory / (SESSION + ".jsonl")
    index = directory / SESSION / "inputs.jsonl"
    events = records(log)
    header, end = events[0], events[-1]
    start = next(row for row in events if row["event"] == "start")

    def seconds(stamp):
        return round((stamp - start["mono_ns"]) / 1e9, 6)

    def wall(stamp):
        ns = header["wall_ns"] + stamp - header["mono_ns"]
        local = datetime.fromtimestamp(ns / 1e9, timezone(timedelta(hours=8)))
        return local.isoformat(timespec="milliseconds")

    fault = next(row for row in events if row["event"] == "target_guard_fault")
    sent = [row for row in events if row["event"] == "prediction" and row["mode"] == "execute"]
    inputs = [row for row in records(index) if row["mode_at_query"] == "execute"]
    by_stamp = {row["stamp"]: row for row in inputs}
    assert len(by_stamp) == len(inputs) == len(sent) + 1
    assert by_stamp[fault["stamp"]]["p"] == fault["prediction"]
    for row in sent:
        frame = by_stamp[row["stamp"]]
        assert frame["p"] == row["p"] and frame["q"] == row["q"]
        assert row["sent_ids"] == [0x155, 0x156, 0x157, 0x159]
    assert end["event"] == "end" and "已收到从臂 STOP 反馈" in end["detail"]
    assert fault["previous_sent"] == sent[-1]["sent"]
    axis, = fault["axes"]
    assert axis["axis"] == "夹爪"
    width = fault["measured_at_send"][6]
    assert math.isclose(width - sent[-1]["sent"][6], axis["gap"])

    checked = sent + [fault]
    fixed_stamps = []
    for row in reversed(checked):
        if row["measured_at_send"][6] != width:
            break
        fixed_stamps.append(row["measured_at_send_ns"]["gripper_ns"])
    joint_gaps = []
    previous = start["initial"]
    for row in checked:
        joint_gaps.append([abs(math.degrees(m - p))
                           for m, p in zip(row["measured_at_send"][:6], previous[:6])])
        if "sent" in row:
            previous = row["sent"]
    crossings = []
    for target in (60, 58, 55, 52, 50):
        hit = next(row for row in sent if row["sent"][6] <= target)
        crossings.append(dict(first_target_at_or_below=target, t=seconds(hit["send_started_ns"]),
                              target=hit["sent"][6], predicted=hit["p"][6], feedback=hit["measured_at_send"][6]))
    phases = [row["plan"]["alignment"]["phase"] for row in inputs]
    changes = [dict(t=seconds(row["stamp"]), phase=phases[i], gripper=row["q"][6], pred=row["p"][6])
               for i, row in enumerate(inputs) if i == 0 or phases[i] != phases[i - 1]]
    corrected = [row for row in inputs if row["plan"]["alignment"]["applied"]]
    ages = [(row["mono_ns"] - row["stamp"]) / 1e6 for row in checked]
    return dict(
        session=SESSION, policy=header["policy"]["version"], bundle_sha256=header["policy"]["sha256"],
        wall_start=wall(start["mono_ns"]), wall_fault=wall(fault["mono_ns"]),
        wall_end=wall(end["mono_ns"]), end=end["detail"],
        command_groups=len(sent), inference_count=len(inputs),
        replans=len({row["plan"]["chunk_stamp"] for row in inputs}),
        phases=dict(Counter(phases)), transitions=changes,
        first_approach_est_mm=corrected[0]["plan"]["alignment"]["distance_mm"],
        last_approach_est_mm=corrected[-1]["plan"]["alignment"]["distance_mm"],
        last_approach_t=seconds(corrected[-1]["stamp"]), target_crossings=crossings,
        fault=dict(t=seconds(fault["mono_ns"]), last_target=sent[-1]["sent"][6], width=width,
                   unsent=fault["prediction"][6], gap=axis["gap"], threshold=axis["tracking_limit"],
                   feedback_age_ms=(fault["mono_ns"] - fixed_stamps[0]) / 1e6),
        stationary_width=dict(start_t=seconds(fixed_stamps[-1]), duration_s=(fixed_stamps[0] - fixed_stamps[-1]) / 1e9,
                              samples=len(fixed_stamps), unique_feedback_stamps=len(set(fixed_stamps))),
        joint_gaps_max_deg=[max(column) for column in zip(*joint_gaps)],
        model_input_to_guard_ms=dict(median=median(ages), max=max(ages)),
        camera_skew_max_ms=max(row["skew_ns"] for row in inputs) / 1e6,
        inputs_sha256=sha256(index.read_bytes()).hexdigest(), run_sha256=sha256(log.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, ensure_ascii=False, allow_nan=False))
