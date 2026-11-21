"""Replay the first eight valid pairs of each held-out B1 episode, entirely from disk."""
from collections import deque
from decimal import Decimal
import json
from pathlib import Path
import shlex
import subprocess
import sys

import cv2
import numpy as np
import torch

work = Path(sys.argv[1])
session = work / "policy_runs" / sys.argv[2]
for entry in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        opts = entry.read_bytes().split(b"\0")
    except OSError:
        continue
    if any(a.endswith(b"/runner.py") and b in (b"infer", b"guard") for a,b in zip(opts,opts[1:])):
        raise RuntimeError("A live policy session is using the host")
initial = next(r["initial"] for line in session.open() if (r:=json.loads(line))["event"] == "start")
desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip())
command = next(s[5:] for s in (desktop/"PiPER-Policy.desktop").read_text().splitlines() if s.startswith("Exec="))
sys.path.insert(0, str(Path(shlex.split(command)[-1]).parent))
from policy import Predictor

torch.set_num_threads(2)
model = Predictor(work, 0)
base = work / "expert_sets/B1_top50_20261120_163349"
split = json.loads((work/"offline_training/b123_goal_v1/split.json").read_text())
results = []
for ep in json.loads((base/"manifest.json").read_text())["episodes"]:
    if split[ep["episode_id"]] != "val":
        continue
    folder = base/ep["episode_dir"]
    rows = [json.loads(s) for s in (folder/"alignment.jsonl").read_text().splitlines()]
    first = ep["first_action_row"]
    views = []
    for role in ("overhead","wrist"):
        files = sorted((p for p in (folder/"episode0/camera/color"/role).iterdir() if p.stem.replace(".","",1).isdigit()),
                       key=lambda p: Decimal(p.stem))
        assert len(files)==len(rows)
        views.append(files)
    histories = {k:deque(maxlen=8) for k in ("expert","closed_grip","field_pose")}
    preds = {}
    for idx in range(first,first+8):
        q=rows[idx]["feedback"]
        expert=q["joints_rad"]+[q["gripper_mm"]]
        images=[cv2.imread(str(v[idx])) for v in views]
        for v in views:
            assert abs(int(Decimal(v[idx].stem)*10**9)-rows[idx]["sample_monotonic_ns"])<=500
        for tag,feedback in (("expert",expert),("closed_grip",[*expert[:6],initial[6]]),("field_pose",initial)):
            model.history=histories[tag]
            preds[tag]=model.predict(images,feedback)
    results.append(dict(recorded=rows[first+7]["target"]["gripper_mm"], **{k:v[6] for k,v in preds.items()}))
assert len(results)==10
stats={}
for tag in results[0]:
    values=np.array([x[tag] for x in results])
    stats[tag]=dict(range_mm=[round(float(a),3) for a in (values.min(),np.median(values),values.max())],
                    positive=int((values>0).sum()))
errors=[abs(r["recorded"]-r["expert"]) for r in results]
print(json.dumps(dict(episodes=len(results),history_frames=8, stats=stats,
    median_matched_grip_error_mm=round(float(np.median(errors)),3))),flush=True)
