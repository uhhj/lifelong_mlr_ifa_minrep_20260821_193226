"""Read an ended B1 run, passive CAN archive and the B1 demonstration metadata."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import median
import struct
import sys


def rows(path):
    with path.open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def units(values):
    return [round(math.degrees(x), 3) for x in values[:6]] + [round(values[6], 3)]


def summary(series):
    return dict(first=series[0], last=series[-1], low=list(map(min, zip(*series))),
                high=list(map(max, zip(*series))))


def triple(values):
    return [round(f(values), 3) for f in (min, median, max)]


work = Path(sys.argv[1])
run_file = work / "policy_runs" / sys.argv[2]
watch = work.parent / "diagnostics" / sys.argv[3]
log = rows(run_file)
start, = [r for r in log if r["event"] == "start"]
end = log[-1]
assert end["event"] == "end"
begin, finish = start["mono_ns"], end["mono_ns"]
assert begin < finish
execution = [r for r in log if r.get("mode") == "execute" and r["event"] == "prediction"]
assert execution and all(begin <= r["mono_ns"] <= finish for r in execution)
anchor = log[0]["wall_ns"] - log[0]["mono_ns"]
bj = timezone(timedelta(hours=8))


def clock(mono):
    return datetime.fromtimestamp((anchor + mono)/1e9, bj).isoformat(timespec="milliseconds")


keys = ["q", "p", "sent", "measured_at_send"]
series = {k: [units(r[k]) for r in execution] for k in keys}
expected = defaultdict(list)
for r in execution:
    s = r["sent"]
    for j in range(3):
        pair = [round(math.degrees(a)*1000) for a in s[2*j:2*j+2]]
        expected[0x155+j].append(struct.pack(">ii", *pair))
    expected[0x159].append(struct.pack(">iHBB", round(s[6]*1000), 1000, 1, 0))

wire = struct.Struct("<QQIBBH8s")
received = defaultdict(list)
on_bus = Counter()
status = {}
grip = []
z_trace = []
feedback_steps = defaultdict(list)
stop = []
errors = 0
motor_enabled = defaultdict(set)
feedback = defaultdict(list)
for part in sorted(watch.glob("frames-*.bin")):
    with part.open("rb") as f:
        size = f.seek(0, 2)
        if size < wire.size:
            continue
        f.seek(0)
        first = wire.unpack(f.read(wire.size))[1]
        f.seek((size // wire.size - 1)*wire.size)
        last = wire.unpack(f.read(wire.size))[1]
        if last < begin or first > finish+300_000_000:
            continue
        f.seek(0)
        for _, t, cid, port, n, flags, data in wire.iter_unpack(f.read(size//wire.size*wire.size)):
            if not begin <= t <= finish+300_000_000 or port:
                continue
            if cid & 0x20000000:
                errors += 1
            if cid in expected:
                if flags & 4:
                    received[cid].append(data)
                else:
                    on_bus[hex(cid)] += 1
            if cid == 0x150:
                stop.append([clock(t), data.hex(), bool(flags & 4)])
            if cid == 0x2a1:
                status.setdefault(data.hex(), clock(t))
            if t > finish or n != 8:
                continue
            if 0x261 <= cid <= 0x266:
                motor_enabled[str(cid-0x260)].add(data[5])
            if 0x2a5 <= cid <= 0x2a7:
                for i, a in enumerate(struct.unpack(">ii", data)):
                    axis = 1+(cid-0x2a5)*2+i
                    if feedback[axis]:
                        feedback_steps[axis].append(abs(feedback[axis][-1] - a/1000))
                    feedback[axis].append(a/1000)
            if cid == 0x2a3:
                z_trace.append((t, struct.unpack_from(">i",data)[0]/1000))
            if cid == 0x2a8:
                grip.append(struct.unpack_from(">i", data)[0]/1000)

base = work / "expert_sets/B1_top50_20261120_163349"
stats = defaultdict(list)
for entry in json.loads((base/"manifest.json").read_text())["episodes"]:
    samples = rows(base/entry["episode_dir"]/"alignment.jsonl")[entry["first_action_row"]:entry["last_action_row"]+1]
    for key in ("feedback", "target"):
        tracks = [units(row[key]["joints_rad"]+[row[key]["gripper_mm"]]) for row in samples]
        stats[key+"_first"].append(tracks[0])
        stats[key+"_peak_grip_mm"].append(max(v[6] for v in tracks))
        stats[key+"_span"].append([max(axis)-min(axis) for axis in zip(*tracks)])
    stats["duration_s"].append((samples[-1]["sample_monotonic_ns"]-samples[0]["sample_monotonic_ns"])/1e9)
assert len(stats["duration_s"]) == 50
demo = {k: [triple(x) for x in zip(*v)] if isinstance(v[0], list) else triple(v) for k,v in stats.items()}
pairs = [r["stamp"] for r in execution]
z0 = z_trace[0][1]
def changes_above_deadband(values, deadband):
    # Count reversals only after excursions of the given size from a local extremum.
    lo = hi = values[0]
    direction = reversals = 0
    for value in values[1:]:
        lo, hi = min(lo, value), max(hi, value)
        if direction <= 0 and value-lo >= deadband:
            reversals += direction == -1
            direction, hi = 1, value
        elif direction >= 0 and hi-value >= deadband:
            reversals += direction == 1
            direction, lo = -1, value
    return reversals

result = dict(start=clock(begin), end=clock(finish), seconds=round((finish-begin)/1e9,3),
              count=len(execution), outcome=end["outcome"], detail=end["detail"], initial=units(start["initial"]),
              max_pair_gap_ms=round(max(y-x for x,y in zip(pairs,pairs[1:]))/1e6,3),
              max_prediction_age_ms=round(max(r["mono_ns"]-r["stamp"] for r in execution)/1e6,3),
              full=all(r["full"] for r in execution), trajectories={k:summary(v) for k,v in series.items()},
              max_target_measurement_error=[round(max(abs(x-y) for x,y in zip(a,b)),3) for a,b in
                  zip(zip(*series["sent"]),zip(*series["measured_at_send"]))],
              predicted_grip_positive=sum(p[-1]>0 for p in series["p"]),
              trajectory_samples=[dict(t_s=round((r["mono_ns"]-begin)/1e9,2),raw=units(r["p"]),
                  measured=units(r["measured_at_send"]),sent=units(r["sent"]))
                  for i,r in enumerate(execution) if i%150==0 or i==len(execution)-1],
              wire_commands={hex(k):dict(count=len(received[k]),match=received[k]==v) for k,v in expected.items()},
              wire_other_targets=dict(on_bus), wire_error_frames=errors, wire_stop=stop, first_seen_status=status,
              wire_feedback_deg={k:triple(v) for k,v in sorted(feedback.items())},
              max_adjacent_feedback_step_deg={k:round(max(v),3) for k,v in feedback_steps.items()},
              max_adjacent_sent_step_deg=[round(max(abs(y-x) for x,y in zip(a,a[1:])),3)
                                          for a in list(zip(*series["sent"]))[:6]],
              reversals_over_2deg={k:[changes_above_deadband(a, 2) for a in list(zip(*series[k]))[:6]]
                                   for k in ("sent","measured_at_send")},
              z_feedback_mm=dict(first=z0, last=z_trace[-1][1], min=min(z for t,z in z_trace), max=max(z for t,z in z_trace)),
              wire_feedback_grip_mm=triple(grip), wire_drive_status={k:sorted(v) for k,v in motor_enabled.items()},
              demonstration_min_median_max=demo)
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
