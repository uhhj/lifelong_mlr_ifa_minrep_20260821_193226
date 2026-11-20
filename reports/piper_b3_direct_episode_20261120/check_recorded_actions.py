"""Re-decode the archived command payloads and screen only the sampled feedback."""
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


def check(episode):
    raw = (episode / "alignment.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    n = len(rows)
    matched = 0
    causal = True
    time = np.array([r["sample_monotonic_ns"] for r in rows], dtype=np.int64)
    feedback = np.rad2deg([r["feedback"]["joints_rad"] for r in rows]) if n else []
    targets = np.full((n, 6), np.nan)
    for i, row in enumerate(rows):
        obs = row["feedback"]
        causal &= max(obs["component_monotonic_ns"]) <= time[i]
        cmd = row["target"]
        if cmd is None:
            continue
        payloads = list(map(bytes.fromhex, cmd["frame_hex"]))
        angles = [v for b in payloads[:3] for v in struct.unpack(">ii", b)]
        width = struct.unpack_from(">i", payloads[3])[0] / 1000
        matched += int(cmd["origin"] == "direct_can_rx"
                       and np.allclose(np.deg2rad(np.array(angles)/1000), cmd["joints_rad"], rtol=0, atol=1e-10)
                       and abs(width - cmd["gripper_mm"]) < 1e-9)
        stamps = cmd["joint_bus_monotonic_ns"] + [cmd["gripper_bus_monotonic_ns"]]
        causal &= max(stamps) == cmd["sample_monotonic_ns"] <= time[i]
        targets[i] = angles
    count = 0
    peak = None
    for i in range(1, n):
        left = np.searchsorted(time, time[i] - 150_000_000, side="right") - 1
        if left < 0 or not np.all(targets[left:i+1] == targets[i]):
            continue
        count += 1
        delta = np.abs(feedback[i] - feedback[i-1])
        axis = int(np.argmax(delta))
        if peak is None or delta[axis] > peak["degrees"]:
            peak = dict(row=i, axis=axis+1, degrees=float(delta[axis]),
                        interval_ms=round((time[i]-time[i-1])/1e6, 3))
    if peak is not None:
        peak["degrees"] = round(peak["degrees"], 3)
    return dict(sha256=hashlib.sha256(raw).hexdigest(), matched=matched,
                causal=bool(causal), held_samples=count, peak=peak)


if __name__ == "__main__":
    base = Path(sys.argv[1]) / "recordings"
    print(json.dumps({name: check(base/name) for name in sys.argv[2:]}))
