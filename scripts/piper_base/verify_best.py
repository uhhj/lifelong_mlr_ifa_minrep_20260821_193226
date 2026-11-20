"""Replay one held-out dual-camera window per task from disk with the saved best policy."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from piper_base import EasyDict, Frontend, GOAL_SHA, HISTORY, Policy, TOKEN_SHA, digest, read_episodes


@torch.no_grad()
def verify(data_root, run_dir):
    torch.set_num_threads(4)
    finished = json.loads((run_dir / "finished.json").read_text())
    best = run_dir / "best.pt"
    assert digest(best) == finished["best_sha256"]
    saved = torch.load(best, map_location="cpu", weights_only=True)
    config = saved["config"]
    assert saved["epoch"] == finished["best_epoch"] and config["script_sha256"] == digest(Path(__file__).with_name("piper_base.py"))
    goal = run_dir / "goal_multitask_model_ep96.pth"
    tokenizer = run_dir / "clip_tokenizer_goal.json"
    assert digest(goal) == GOAL_SHA and digest(tokenizer) == TOKEN_SHA
    with torch.serialization.safe_globals([(EasyDict, "easydict.EasyDict")]):
        checkpoint = torch.load(goal, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k,v in checkpoint["state_dict"].items()}
    front = Frontend(state, config["lora_scale"], Tokenizer.from_file(str(tokenizer))).cuda()
    model = Policy(state, front.language.cpu(), config["normalization"])
    model.load_state_dict(saved["policy"], strict=True)
    model.cuda().eval()
    episodes, _ = read_episodes(data_root)
    samples = []
    for task in range(3):
        ep = next(e for e in episodes if e["task"] == task and e["split"] == "val")
        row = len(ep["q"]) // 2
        ids = list(range(row-HISTORY+1, row+1))
        vision = front.encode(ep["cameras"],ids,task)[None].cuda()
        q = torch.from_numpy(ep["q"][ids])[None].cuda()
        tag = torch.tensor([task], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, pred = model(vision,q,tag)
        assert pred.shape == (1,7) and torch.isfinite(pred).all()
        error = (pred.cpu().numpy()[0] - ep["a"][row])
        samples.append(dict(task=f"B{task+1}", episode=ep["id"], valid_relative_row=row,
                            joint_mae_deg=float(np.abs(np.rad2deg(error[:6])).mean()),
                            gripper_abs_error_mm=float(np.abs(error[6]))))
    return dict(best_epoch=saved["epoch"], best_sha256=finished["best_sha256"], from_raw_frames=samples)


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()), allow_nan=False))
