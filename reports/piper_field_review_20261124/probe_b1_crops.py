"""Paired crop sensitivity check on archived successful B1 episodes outside the 50 used for training."""
import hashlib
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class HorizontalCrop(nn.Module):
    def __init__(self, offsets):
        super().__init__()
        self.offsets = offsets

    def forward(self, frames):
        assert len(frames) == len(self.offsets)
        return torch.stack([frame[:, 8:120, x:x+112] for frame, x in zip(frames, self.offsets)])


@torch.inference_mode()
def replay(work):
    root = work / "offline_diagnostics/b1_ode_lerobot_20261123"
    sys.path.insert(0, str(root / "code"))
    from ode_policy_adapter import build_policy
    saved = torch.load(root / "run/epoch_50.pt", map_location="cpu", weights_only=True)
    manifest = (root / "selection.json").read_bytes()
    assert saved["epoch"] == 50 and saved["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    policy = build_policy(saved["stats"], saved["config"])
    policy.load_state_dict(saved["ema"], strict=True)
    policy.to(device).eval()
    selection = json.loads(manifest)
    selected = set(selection["selected_ids"])
    records = [r for r in selection["ranking"] if r["id"] not in selected]
    transforms = policy.obs_encoder.key_transform_map
    roles = ("overhead", "wrist")
    standard = {role:transforms[role][1] for role in roles}
    center = [8]*8
    jitter = [4,12,4,12,4,12,8,8]
    conditions = {"standard":{}, "center_control":{"overhead":center},
                  "overhead_left_4":{"overhead":[4]*8}, "overhead_right_4":{"overhead":[12]*8},
                  "overhead_history_jitter":{"overhead":jitter}, "wrist_history_jitter":{"wrist":jitter}}
    errors, differences = defaultdict(list), defaultdict(list)
    seed_differences, shift_pixels = defaultdict(list), defaultdict(list)
    previous_window = {}
    previous_feature, feature_change = {}, defaultdict(list)
    used_episodes = 0
    window_counts = defaultdict(int)
    for rec in records:
        folder = work / "batches/B1_20261123" / rec["source"]
        raw = (folder / "alignment.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == rec["alignment_sha256"]
        rows = [json.loads(line) for line in raw.splitlines()][rec["first"]:]
        def vectors(key):
            return np.array([r[key]["joints_rad"]+[r[key]["gripper_mm"]] for r in rows],np.float32)
        state, cmd = vectors("feedback"), vectors("target")
        if cmd[0,6] <= 55:
            continue
        crosses = np.flatnonzero((cmd[:-1,6]>55) & (cmd[1:,6]<=55))+1
        if not len(crosses):
            continue
        indices = {name:int(crosses[0])-offset for name,offset in (("before_close",18),("closing",3))}
        indices = {name:j for name,j in indices.items() if 7 <= j <= len(rows)-8}
        if not indices:
            continue
        used_episodes += 1
        files = {role:sorted((folder / "episode0/camera/color" / role).glob("*.jpg"),
                            key=lambda f:Decimal(f.stem))[rec["first"]:] for role in roles}
        for phase, j in indices.items():
            window_counts[phase] += 1
            obs = {"agent_pos":torch.from_numpy(state[None,j-7:j+1]).to(device)}
            for role in roles:
                imgs = []
                for f in files[role][j-7:j+1]:
                    frame = cv2.imread(str(f))
                    padded = cv2.copyMakeBorder(frame,80,80,0,0,cv2.BORDER_CONSTANT,value=0)
                    rgb = cv2.cvtColor(cv2.resize(padded,(128,128),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                    imgs.append(rgb.transpose(2,0,1))
                obs[role] = torch.from_numpy(np.stack(imgs)[None]).to(device).float().div_(255)
            anchor = torch.from_numpy(cmd[None,j-1]).to(device)
            expected = cmd[j:j+8]
            for role in roles:
                transforms[role][1] = standard[role]
            feature = {}
            def record(role):
                def hook(module, args, result):
                    feature[role] = result.detach()
                return hook
            hooks = [policy.obs_encoder.key_model_map[role].register_forward_hook(record(role)) for role in roles]
            torch.manual_seed(42)
            try:
                seeded_ref = policy.predict_action(obs,a0=anchor,eps_center=True)["action"][0].cpu().numpy()
            finally:
                for handle in hooks:
                    handle.remove()
            for name, custom in conditions.items():
                for role in roles:
                    transforms[role][1] = HorizontalCrop(custom[role]) if role in custom else standard[role]
                pred = policy.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))["action"][0].cpu().numpy()
                if name == "standard":
                    reference = pred.copy()
                errors[name].append(np.abs(pred-expected))
                differences[name].append(np.abs(pred-reference))
            for role in roles:
                transforms[role][1] = standard[role]
            if phase in previous_window:
                for role in roles:
                    name = role+"_from_other_episode"
                    changed = dict(obs)
                    changed[role] = previous_window[phase][role]
                    shift_pixels[name].append(float((changed[role]-obs[role]).abs().mean()*255))
                    first = feature[role]
                    second = previous_feature[phase][role]
                    feature_change[role].append(float(torch.linalg.vector_norm(first-second) /
                                                        torch.linalg.vector_norm(first).clamp_min(1e-9)))
                    mean_pred = policy.predict_action(changed,a0=anchor,eps=torch.zeros_like(anchor))["action"][0].cpu().numpy()
                    torch.manual_seed(42)
                    seed_pred = policy.predict_action(changed,a0=anchor,eps_center=True)["action"][0].cpu().numpy()
                    differences[name].append(np.abs(mean_pred-reference))
                    seed_differences[name].append(np.abs(seed_pred-seeded_ref))
                    errors[name].append(np.abs(mean_pred-expected))
            previous_window[phase] = {r:obs[r] for r in roles}
            previous_feature[phase] = feature
    def summarize(results):
        table = {}
        for name, windows in results.items():
            arr = np.stack(windows)
            deg = np.rad2deg(arr[:,:,:6])
            table[name] = dict(windows=len(windows),joint_mean_deg=float(deg.mean()),grip_mean_mm=float(arr[:,:,6].mean()),
                               window_peak_joint_deg_p50_p95=np.percentile(deg.max((1,2)),[50,95]).tolist(),
                               window_peak_grip_mm_p50_p95=np.percentile(arr[:,:,6].max(1),[50,95]).tolist())
        return table
    assert np.max(np.stack(differences["center_control"])) < 1e-5
    return dict(nontraining_ranked_episodes=len(records), evaluated_episodes=used_episodes, windows=dict(window_counts),
                horizon=8, variant_minus_standard=summarize(differences), absolute_expert_error=summarize(errors),
                sampled_seed42_variant_minus_standard=summarize(seed_differences),
                feature_relative_L2_p50_p95={role:np.percentile(v,[50,95]).tolist() for role,v in feature_change.items()},
                cross_episode_mean_pixel_change_0_to_255={k:float(np.mean(v)) for k,v in shift_pixels.items()})


if __name__ == "__main__":
    cv2.setNumThreads(1)
    print(json.dumps(replay(Path(sys.argv[1])), indent=2))
