"""Inspect B1 training labels at shoulder extension and at the first closing command."""
import json
from pathlib import Path
import random
import sys

import numpy as np

from compare_b1_pickup_phase import degrees_mm, expert_episodes


def audit(expert_source, field_dir):
    episodes = expert_episodes(expert_source)
    assert len(episodes) == 50 and len({e["id"] for e in episodes}) == 50
    validation = set(random.Random(42).sample(sorted(e["id"] for e in episodes), 10))
    first_commands = []
    episode_medians = []
    sample_count = 0
    for episode in episodes:
        if episode["id"] in validation:
            continue
        q, a = (degrees_mm(episode[key]) for key in ("q", "a"))
        if min(q[0, 6], a[0, 6]) <= 50:
            continue
        t = (np.array(episode["t"]) - episode["t"][0]) / 1e9
        assert np.all(np.diff(t) > 0)
        first_target = np.flatnonzero(a[:, 6] <= 50)[0]
        first_feedback = np.flatnonzero(q[:, 6] <= 50)[0]
        first_commands.append(q[first_target, 1])
        start_angle = np.interp(t - .2, t, q[:, 1])
        extending = (q[:, 1] - start_angle) / .2 >= 10
        eligible = (t >= .2) & extending & (t + .3 < t[first_feedback])
        if np.any(eligible):
            future_command = np.interp(t + .3, t, a[:, 1])
            future_feedback = np.interp(t + .3, t, q[:, 1])
            episode_medians.append([np.median((target - q[:, 1])[eligible])
                                    for target in (a[:, 1], future_command, future_feedback)])
            sample_count += int(eligible.sum())

    records = [json.loads(line) for line in (field_dir/"main.jsonl").read_bytes().splitlines()]
    assert records[-1]["event"] == "end"
    start = next(row["mono_ns"] for row in records if row["event"] == "start")
    run = [row for row in records if row.get("mode") == "execute" and row["event"] == "prediction"]
    assert run[0]["q"][6] > 50 and run[0]["sent"][6] > 50
    first = next(row for row in run if row["sent"][6] <= 50)
    return dict(train=40, comparable_closures=len(first_commands),
                expert_j2_at_first_close_command_deg=np.round(np.percentile(first_commands,[0,50,100]),3).tolist(),
                field_first_close=dict(input_time_s=round((first["stamp"]-start)/1e9,3),
                    j2_input_deg=round(float(np.degrees(first["q"][1])),3),
                    gripper_raw_mm=round(first["p"][6],3),gripper_sent_mm=round(first["sent"][6],3)),
                approach_episodes=len(episode_medians),approach_frames=sample_count,
                median_j2_deg_current_target_future_target_future_feedback=np.round(np.median(episode_medians,axis=0),3).tolist())


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, allow_nan=False))
