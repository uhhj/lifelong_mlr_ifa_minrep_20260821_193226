"""Compare the saved camera inputs and timestamps for the two archived B1 executions."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf8")]


def check(folder):
    events = read(folder.with_suffix(".jsonl"))
    rows = [r for r in events if r["event"] == "prediction" and r.get("mode") == "execute"]
    start = next(r["mono_ns"] for r in events if r["event"] == "start")
    end = events[-1]["mono_ns"]
    all_inputs = read(folder / "inputs.jsonl")
    index = {r["stamp"]:r for r in all_inputs}
    offsets = Counter()
    delays = []
    feedback_age = []
    camera_age = []
    pixel_error = []
    wrong_view = []
    wrong_channel = []
    for r in all_inputs:
        if not start <= r["stamp"] <= end:
            continue
        # The scene is JPEG and the model input is PNG. Decode both in OpenCV channel order.
        full = cv2.imread(str(folder/r["images"]["scene"]))
        original = []
        for half in np.hsplit(full, 2):
            padded = cv2.copyMakeBorder(half,80,80,0,0,cv2.BORDER_CONSTANT,value=0)
            original.append(cv2.resize(padded,(128,128),interpolation=cv2.INTER_AREA))
        saved = cv2.imread(str(folder/r["images"]["model"]))
        def difference(image):
            return float(np.abs(image.astype(np.int16)-saved).mean())
        same = np.hstack(original)
        pixel_error.append(difference(same))
        wrong_view.append(difference(np.hstack(original[::-1])))
        wrong_channel.append(difference(same[:,:,::-1]))
    for row in rows:
        r = index[row["stamp"]]
        stamp = row["send_started_ns"]
        offsets[r["plan"]["offset"]] += 1
        delays.append((stamp-r["plan"]["chunk_stamp"])/1e6)
        feedback_age.append((r["stamp"]-min(r["feedback_ns"].values()))/1e6)
        camera_age.append((stamp-r["stamp"])/1e6)
    def quant(values):
        return np.round(np.percentile(values,[50,95,100]),4).tolist()
    return dict(active_inputs=len(pixel_error), dispatched=len(rows), plan_offsets=dict(offsets),
                plan_age_ms_p50_p95_max=quant(delays), frame_age_ms_p50_p95_max=quant(camera_age),
                feedback_age_ms_p50_p95_max=quant(feedback_age), plan_at_least_200ms=int(np.sum(np.array(delays)>=200)),
                pixel_mae_mean_0_to_255={"as_saved":float(np.mean(pixel_error)),
                                         "swap_views":float(np.mean(wrong_view)), "swap_R_B":float(np.mean(wrong_channel))})


def replay_field(work, folders):
    """Replay the actual RNG stream, then perturb only the images at each original plan."""
    import torch
    root = work / "offline_diagnostics/b1_ode_lerobot_20261123"
    sys.path.insert(0, str(root / "code"))
    from ode_policy_adapter import build_policy
    saved = torch.load(root/"run/epoch_50.pt",map_location="cpu",weights_only=True)
    policy = build_policy(saved["stats"],saved["config"])
    policy.load_state_dict(saved["ema"], strict=True)
    policy.cuda().eval()
    result = []
    with torch.inference_mode():
        for folder,(cx,cy) in zip(folders,[(248,350),(218,357)]):
            x,y = cx//5,(cy+80)//5
            inputs = read(folder/"inputs.jsonl")
            index = {r["stamp"]:r for r in inputs}
            active = [r for r in inputs if r["mode_at_query"]=="execute" and r["plan"]["ready"]]
            errors = {k:[] for k in ("baseline_vs_log", "mask_wood_area", "gray_overhead", "gray_wrist")}
            counts = 0
            torch.manual_seed(42)
            for r in active:
                plan = r["plan"]
                if plan["chunk_stamp"] == r["stamp"]:
                    counts += 1
                    history = [index[s] for s in r["history"]]
                    pairs = [cv2.cvtColor(cv2.imread(str(folder/h["images"]["model"])),cv2.COLOR_BGR2RGB) for h in history]
                    rgb = torch.from_numpy(np.stack(pairs)).cuda().permute(0,3,1,2).contiguous().float().div_(255)
                    obs = {"overhead":rgb[None,:,:,:,:128].contiguous(),"wrist":rgb[None,:,:,:,128:].contiguous(),
                           "agent_pos":torch.tensor([[h["q"] for h in history]],dtype=torch.float32,device="cuda")}
                    anchor = torch.tensor([plan["anchor"]],dtype=torch.float32,device="cuda")
                    before = torch.cuda.get_rng_state()
                    preds = {}
                    for variant in ("baseline", "mask_wood_area", "gray_overhead", "gray_wrist"):
                        changed = dict(obs)
                        if variant == "mask_wood_area":
                            img = obs["overhead"].clone()
                            img[:,:,:,y-7:y+8,x-7:x+8] = .5
                            changed["overhead"] = img
                        elif variant.startswith("gray_"):
                            role = variant.removeprefix("gray_")
                            changed[role] = torch.full_like(obs[role],.5)
                        torch.cuda.set_rng_state(before)
                        preds[variant] = policy.predict_action(changed,a0=anchor,eps_center=True)["action"][0].cpu().numpy()
                        if variant == "baseline":
                            after = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(after)
                t = plan["offset"]
                base = preds["baseline"][t]
                errors["baseline_vs_log"].append(np.abs(base-np.asarray(r["p"])))
                for variant in ("mask_wood_area","gray_overhead","gray_wrist"):
                    errors[variant].append(np.abs(preds[variant][t]-base))
            numbers = {}
            for variant, values in errors.items():
                a = np.stack(values)
                numbers[variant] = dict(max_joint_deg=float(np.rad2deg(a[:,:6]).max()),max_grip_mm=float(a[:,6].max()),
                                        joint_mean_deg=float(np.rad2deg(a[:,:6]).mean()),grip_mean_mm=float(a[:,6].mean()))
            result.append(dict(predictions=len(active),plans=counts,delta=numbers))
    return result


if __name__ == "__main__":
    cv2.setNumThreads(1)
    root = Path(sys.argv[1])/"policy_runs"
    names = ["20261123_234228_1446439_B1-ode50-20261123","20261123_234953_1448999_B1-ode50-20261123"]
    folders = [root/name for name in names]
    print(json.dumps({"dispatch":[check(folder) for folder in folders],
                      "frozen_replay":replay_field(root.parent,folders)}, indent=2))
