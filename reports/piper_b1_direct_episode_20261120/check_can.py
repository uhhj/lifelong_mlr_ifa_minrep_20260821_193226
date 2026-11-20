"""Verify direct-mode action and observation samples against archived can0 frames."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


RECORD = np.dtype([("wall", "<u8"), ("mono", "<u8"), ("cid", "<u4"),
                   ("port", "u1"), ("size", "u1"), ("flags", "<u2"), ("data", "V8")])
TARGET = (0x155, 0x156, 0x157, 0x159)
FEEDBACK = (0x2A5, 0x2A6, 0x2A7, 0x2A8)


def check(episode, segments):
    sidecar = (episode / "alignment.jsonl").read_bytes()
    rows = [json.loads(line) for line in sidecar.splitlines()]
    start, end = (rows[i]["sample_monotonic_ns"] for i in (0, -1))
    indexed = {}
    occurrences = defaultdict(list)
    for path in segments:
        size = path.stat().st_size // RECORD.itemsize
        if not size:
            continue
        raw = np.memmap(path, dtype=RECORD, mode="r", shape=(size,))
        lo, hi = raw["mono"].searchsorted([start - 100_000_000, end + 100_000_001])
        for frame in raw[lo:hi]:
            cid = int(frame["cid"])
            if frame["port"] or cid not in TARGET + FEEDBACK or frame["size"] != 8:
                continue
            stamp = int(frame["wall"])
            data = frame["data"].tobytes()
            indexed[cid, stamp] = data
            occurrences[cid, data].append(stamp)
    offsets = {}
    for row in rows:
        target = row["target"]
        if target is None:
            continue
        for cid, ts, data in zip(TARGET[:3], target["joint_bus_monotonic_ns"], target["frame_hex"]):
            times = occurrences[cid, bytes.fromhex(data)]
            if len(times) == 1:
                offsets[cid, ts] = times[0] - ts
    if not offsets or len(set(offsets.values())) != 1:
        raise ValueError("The unique command frames do not establish a single clock offset")
    offset = next(iter(offsets.values()))
    counts = Counter()
    for row in rows:
        target, feedback = row["target"], row["feedback"]
        if target is not None:
            stamps = target["joint_bus_monotonic_ns"] + [target["gripper_bus_monotonic_ns"]]
            raw = [indexed.get((cid, ts + offset)) for cid, ts in zip(TARGET, stamps)]
            counts["target_present"] += 1
            counts["target_exact"] += raw == list(map(bytes.fromhex, target["frame_hex"]))
        if feedback is not None:
            counts["feedback_present"] += 1
            raw = [indexed.get((cid, ts + offset)) for cid, ts in zip(FEEDBACK, feedback["component_monotonic_ns"])]
            if all(raw):
                joints = [v for data in raw[:3] for v in struct.unpack(">ii", data)]
                radians = np.deg2rad(np.asarray(joints) / 1000)
                width = struct.unpack_from(">i", raw[3])[0] / 1000
                if np.allclose(radians, feedback["joints_rad"], rtol=0, atol=1e-10) and abs(width-feedback["gripper_mm"]) < 1e-9:
                    counts["feedback_exact"] += 1
    return dict(rows=len(rows), anchors=len(offsets), alignment_sha256=hashlib.sha256(sidecar).hexdigest(), **counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("segments", nargs="+", type=Path, help="Passive segments covering the episode on the same host boot")
    args = parser.parse_args()
    print(json.dumps(check(args.episode, args.segments)))
