"""Replay lateral plans from archived inputs and the deployment's offline math module."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def replay(root, name, flange):
    runs = root / "policy_runs"
    events = rows(runs / (name + ".jsonl"))
    manifest = events[0]["policy"]["alignment"]
    algorithm = root / manifest["algorithm"]
    weights = root / manifest["weights"]
    policy = weights / "private_policy.npz"
    settle_weights = weights / "private_settle.npz"
    for file in (algorithm, policy, settle_weights):
        assert sha256(file.read_bytes()).hexdigest() == manifest["files"][file.relative_to(root).as_posix()]
    spec = spec_from_file_location("grounded_audit", algorithm)
    solver = module_from_spec(spec)
    spec.loader.exec_module(solver)
    with np.load(policy) as archive:
        parameters = {k: archive[k] for k in ("matrix", "tip", "gain", "cap")}
    with np.load(settle_weights) as archive:
        settle = dict(archive)

    def jaw_xy(q):
        transform = flange(np.rad2deg(np.asarray(q, dtype=float)[..., :6]))
        return (transform[..., :3, 3] + transform[..., :3, :3] @ parameters["tip"])[..., :2]

    start = next(e["mono_ns"] for e in events if e["event"] == "start")
    inputs = rows(runs / name / "inputs.jsonl")
    by_stamp = {r["stamp"]: r for r in inputs}
    chunks = {}
    for row in inputs:
        if row["mode_at_query"] != "execute":
            continue
        plan = row["plan"]
        alignment = plan["alignment"]
        if not alignment["applied"]:
            continue
        stamp = plan["chunk_stamp"]
        if stamp not in chunks:
            anchor = by_stamp[stamp]
            history = np.array([by_stamp[t]["q"] for t in anchor["history"]])
            stereo = cv2.imread(str(runs / name / anchor["images"]["scene"]))
            assert stereo.shape == (480, 1280, 3)
            uv = np.array(alignment["block_uv"])
            raw = np.r_[uv, 1.] @ parameters["matrix"]
            origin = jaw_xy(history[-1])
            distance = np.linalg.norm(raw - origin)
            assert abs(distance - alignment["distance_mm"]) < 0.03
            goal = solver.settle_goal(uv, history, stereo[:, 640:], parameters, settle, flange)
            endpoint = solver.move(parameters, origin, goal, "feedback", "heatmap")
            stats = {
                "seconds": (stamp - start) / 1e9,
                "endpoint_change_in_raw_distance_mm": float(np.linalg.norm(raw - endpoint) - distance),
                "step_norm_mm": float(np.linalg.norm(endpoint - origin)),
                "max_reconstruction_error_mm": 0.0,
            }
            chunks[stamp] = origin, endpoint, stats
        origin, endpoint, stats = chunks[stamp]
        expected = origin + (plan["offset"] + 1) / 8 * (endpoint - origin)
        error = float(np.linalg.norm(jaw_xy(row["p"]) - expected))
        stats["max_reconstruction_error_mm"] = max(stats["max_reconstruction_error_mm"], error)
    return {"name": name, "plans": [stats for _, _, stats in chunks.values()]}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root / "offline_diagnostics" / "b1_spatial_ablation_20261121_2034"))
    from fit_gripper import flange

    print(json.dumps({"runs": [replay(root, name, flange) for name in sys.argv[2:]]}, indent=2, allow_nan=False))
