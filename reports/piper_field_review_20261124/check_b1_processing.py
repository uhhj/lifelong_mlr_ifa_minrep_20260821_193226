"""Read original B1 rows, compressed images and the finalized ODE cache. Print aggregate evidence."""
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np


def percentiles(values):
    return np.round(np.percentile(values, [0, 50, 95, 99, 100]), 4).tolist()


def raw_vector(row):
    return np.asarray([*row["joints_rad"], row["gripper_mm"]], dtype=np.float32)


def check(work):
    directory = work / "offline_diagnostics/b1_ode_lerobot_20261123"
    cache = directory / "ode_cache"
    selection = json.loads((directory / "selection.json").read_bytes())
    metadata = json.loads((cache / "metadata.json").read_bytes())
    source = {r["id"]: r for r in selection["ranking"]}
    assert {r["id"] for r in metadata["episodes"]} == set(selection["selected_ids"])
    stats = json.loads((directory / "run/protocol.json").read_bytes())["stats"]
    arrays = {n: np.load(cache / (n + ".npy"), mmap_mode="r") for n in
              ("state", "action", "command", "head_camera", "wrist_camera", "stamps_ns", "episode_ends")}
    issues = Counter()
    count = Counter()
    timing = defaultdict(list)
    old_labels = Counter()
    old_episodes = set()
    jumps_deg, jumps_mm = [], []
    limit = np.zeros((2, 2))
    starts = []
    for i, ep in enumerate(metadata["episodes"]):
        start, stop = ep["start"], ep["stop"]
        starts.append(start)
        assert (start == (0 if i == 0 else arrays["episode_ends"][i-1])
                and stop == arrays["episode_ends"][i])
        record = source[ep["id"]]
        base = work / "batches/B1_20261123" / record["source"]
        alignment = (base / "alignment.jsonl").read_bytes()
        assert hashlib.sha256(alignment).hexdigest() == record["alignment_sha256"]
        rows = [json.loads(line) for line in alignment.splitlines()][record["first"]:]
        assert len(rows) == stop-start
        previous = None
        last_was_old = False
        width = np.array([r["target"]["gripper_mm"] for r in rows])
        crossings = np.flatnonzero((width[:-1]>55) & (width[1:]<=55))+1
        first_close = int(crossings[0]) if width[0]>55 and len(crossings) else None
        images = {}
        for role in ("overhead", "wrist"):
            files = (base / "episode0/camera/color" / role).glob("*.jpg")
            images[role] = sorted(files, key=lambda f: Decimal(f.stem))[record["first"]:]
            assert len(images[role]) == len(rows)
        for j, r in enumerate(rows):
            k = start+j
            ts = r["sample_monotonic_ns"]
            feedback, target = r["feedback"], r["target"]
            state, command = raw_vector(feedback), raw_vector(target)
            words = [bytes.fromhex(word) for word in target["frame_hex"]]
            joints = np.deg2rad(np.array(struct.unpack(">6i", b"".join(words[:3]))) / 1000)
            wire = np.r_[joints, struct.unpack(">i", words[3][:4])[0]/1000].astype(np.float32)
            for name, a, b in [
                ("CAN_to_command", wire, command), ("feedback_to_state", state, arrays["state"][k]),
                ("target_to_command", command, arrays["command"][k]),
                ("previous_command_anchor", state if j == 0 else previous, arrays["action"][k]),
                ("timestamp", ts, int(arrays["stamps_ns"][k]))]:
                issues[name] += int(not np.array_equal(a, b))
            previous = command
            obs = [r["cameras"][key]["monotonic_ns"] for key in images]
            issues["pair_clock"] += int(ts != max(obs))
            timing["view_skew_ms"].append(abs(obs[0]-obs[1])/1e6)
            timing["oldest_feedback_age_ms"].append((ts-min(feedback["component_monotonic_ns"]))/1e6)
            timing["oldest_joint_command_age_ms"].append((ts-min(target["joint_bus_monotonic_ns"]))/1e6)
            timing["grip_command_age_ms"].append((ts-target["gripper_bus_monotonic_ns"])/1e6)
            old = timing["oldest_joint_command_age_ms"][-1] > 100
            if old:
                old_episodes.add(ep["id"])
                old_labels["rows"] += 1
                old_labels["bursts"] += int(not last_was_old)
                old_labels["first_half_second"] += int(ts-rows[0]["sample_monotonic_ns"] < 500_000_000)
                old_labels["last_half_second"] += int(rows[-1]["sample_monotonic_ns"]-ts < 500_000_000)
                if first_close is not None:
                    old_labels["within_half_second_of_first_close"] += int(abs(ts-rows[first_close]["sample_monotonic_ns"]) <= 500_000_000)
                    old_labels["before_first_close"] += int(j < first_close)
                if j:
                    old_labels["identical_target_to_previous_row"] += int(np.array_equal(command, arrays["command"][k-1]))
                    delta = state-arrays["state"][k-1]
                    old_labels["feedback_motion_gt_0p25deg_or_1mm_in_one_frame"] += int(
                        np.rad2deg(np.abs(delta[:6])).max()>.25 or abs(delta[6])>1)
            if last_was_old and not old:
                change = command-arrays["command"][k-1]
                jumps_deg.append(float(np.rad2deg(np.abs(change[:6])).max()))
                jumps_mm.append(float(abs(change[6])))
            last_was_old = old
            if j:
                timing["frame_gap_ms"].append((ts-rows[j-1]["sample_monotonic_ns"])/1e6)
            for role, name in (("overhead","head_camera"), ("wrist","wrist_camera")):
                path = images[role][j]
                issues["image_file_timestamp"] += int(abs(int(Decimal(path.stem)*10**9)-ts)>500)
                image = cv2.imread(str(path))
                assert image.shape == (480, 640, 3)
                padded = cv2.copyMakeBorder(image, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=0)
                small = cv2.resize(padded, (128,128), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2,0,1)
                issues[name] += int(not np.array_equal(rgb, arrays[name][k]))
                count["camera_images"] += 1
            count["rows"] += 1
        count["episodes"] += 1
    for i, (key, values) in enumerate((("agent_pos", arrays["state"]), ("action", np.concatenate(
            [arrays["command"], arrays["action"][starts]])))):
        low, high = [np.array(stats[key][field],dtype=np.float32) for field in ("min","max")]
        issues["stats_limits"] += int(not (np.array_equal(values.min(0), low)
                                           and np.array_equal(values.max(0), high)))
        scale = 2 / (high-low)
        offset = -1-scale*low
        err = np.abs(((values*scale+offset)-offset)/scale - values)
        limit[i] = [np.rad2deg(err[:,:6]).max(), err[:,6].max()]
    return dict(counts=dict(count), mismatches=dict(issues),
                ms_min_p50_p95_p99_max={k:percentiles(v) for k,v in timing.items()},
                age_over_100ms={k:int(np.count_nonzero(np.asarray(v)>100)) for k,v in timing.items()},
                normalization_roundtrip_max_deg_mm=limit.tolist(), held_command_over_100ms={
                    **dict(old_labels), "episodes":len(old_episodes), "bursts_with_next_fresh_row":len(jumps_deg),
                    "next_target_joint_step_deg_min_p50_p95_p99_max":percentiles(jumps_deg) if jumps_deg else [],
                    "next_target_grip_step_mm_min_p50_p95_p99_max":percentiles(jumps_mm) if jumps_mm else []})


if __name__ == "__main__":
    cv2.setNumThreads(1)
    print(json.dumps(check(Path(sys.argv[1])), indent=2))
