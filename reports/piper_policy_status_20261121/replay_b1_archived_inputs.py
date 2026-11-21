"""Recompute saved field predictions and exchange archived image/state histories.

Only saved RGB, state, deployment code and weights are used. Arguments are the
dataset root and JSON mapping expert IDs to (field stamp, expert action row).
"""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import cv2
import numpy as np
import torch


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def main(work, plan):
    episode = "20261121_171512_411687_B1"
    folder = work/"policy_runs"/episode
    log = rows(folder.with_suffix(".jsonl"))
    assert log[-1]["event"] == "end"
    start = next(r["mono_ns"] for r in log if r["event"] == "start")
    archive = {r["stamp"]:r for r in rows(folder/"inputs.jsonl")}
    active = [r for r in archive.values() if start<=r["stamp"]<=log[-1]["mono_ns"]]
    assert len(active)==176
    # Resolve the same installed policy module without starting its desktop launcher.
    desktop = Path(subprocess.check_output(["xdg-user-dir","DESKTOP"],text=True).strip())
    command = next(s[5:] for s in (desktop/"PiPER-Policy.desktop").read_text().splitlines() if s.startswith("Exec="))
    sys.path.insert(0,str(Path(shlex.split(command)[-1]).parent))
    from policy import Predictor, prepare_views
    torch.set_num_threads(2)
    model = Predictor(work,0)

    def encode(pixels):
        front = model.front
        x = torch.from_numpy(pixels).cuda().permute(0,3,1,2).float()/255
        features = front.vision(pixel_values=(x-front.mean)/front.std).pooler_output
        language = front.language[0]
        return (front.proj(features)*(1+front.scale(language))+front.shift(language)).half()

    def predict(v,q):
        _,p = model.model(v[None],q[None],model.tag)
        return p[0]

    def collect(v,q):
        residual = predict(v,q)-q[-1]
        return np.degrees(residual[[0,4]].float().cpu().numpy()).round(5).tolist()

    needed = sorted({stamp for r in active for stamp in r["history"]})
    field_v, field_q = {}, {}
    numbers = []
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for stamp in needed:
            r = archive[stamp]
            rgb = cv2.cvtColor(cv2.imread(str(folder/r["images"]["model"])),cv2.COLOR_BGR2RGB)
            assert rgb.shape==(224,448,3)
            field_v[stamp] = encode(np.stack(np.split(rgb,2,axis=1)))
            field_q[stamp] = torch.tensor(r["q"],dtype=torch.float32,device="cuda")
        def field_history(r):
            return tuple(torch.stack([series[s] for s in r["history"]]) for series in (field_v,field_q))
        for r in active:
            p = predict(*field_history(r)).float().cpu().numpy()
            numbers.append(np.abs(np.array(r["p"])-p))
        diff = np.max(numbers,axis=0)
        assert np.degrees(max(diff[:6]))<.1 and diff[6]<.1, diff
        base = work/"expert_sets/B1_top50_20261120_163349"
        raw_manifest = (base/"manifest.json").read_bytes()
        assert hashlib.sha256(raw_manifest).hexdigest()=="10fc15aff94bc6deb746c4f838ad04fef8156cddaea153b49301376aec3c38cb"
        entries = {e["episode_id"]:e for e in json.loads(raw_manifest)["episodes"]}
        results = []
        for ep,pairs in plan.items():
            entry = entries[ep]
            source = base/entry["episode_dir"]
            raw = (source/"alignment.jsonl").read_bytes()
            assert hashlib.sha256(raw).hexdigest()==entry["alignment_sha256"]
            expert = [json.loads(s) for s in raw.splitlines()][entry["first_action_row"]:entry["last_action_row"]+1]
            cameras = []
            for cam in ("overhead","wrist"):
                paths=sorted((p for p in (source/"episode0/camera/color"/cam).iterdir()
                              if p.stem.replace(".","",1).isdigit()),key=lambda p:Decimal(p.stem))
                cameras.append(paths[entry["first_action_row"]:entry["last_action_row"]+1])
            expert_v,expert_q = {},{}
            for idx in sorted({max(i,0) for _,j in pairs for i in range(j-7,j+1)}):
                row=expert[idx]
                assert all(abs(int(Decimal(cam[idx].stem)*10**9)-row["sample_monotonic_ns"])<=500 for cam in cameras)
                expert_v[idx]=encode(prepare_views([cv2.imread(str(cam[idx])) for cam in cameras]))
                q=row["feedback"]
                expert_q[idx]=torch.tensor(q["joints_rad"]+[q["gripper_mm"]],dtype=torch.float32,device="cuda")
            for stamp,j in pairs:
                v,q=field_history(archive[stamp])
                idx=[max(i,0) for i in range(j-7,j+1)]
                ev,eq=(torch.stack([series[i] for i in idx]) for series in (expert_v,expert_q))
                actual=np.array(expert[j]["target"]["joints_rad"])-np.array(expert[j]["feedback"]["joints_rad"])
                results.append(dict(t=round((stamp-start)/1e9,3),expert=ep,
                    field=collect(v,q),expert_views=collect(ev,q),expert_state=collect(v,eq),
                    expert_overhead=collect(torch.stack((ev[:,0],v[:,1]),1),q),
                    expert_wrist=collect(torch.stack((v[:,0],ev[:,1]),1),q),
                    all_expert=collect(ev,eq),recorded_expert=np.degrees(actual[[0,4]]).round(5).tolist()))
    return dict(verified_inputs=len(numbers),max_replay_errors_deg_mm=[float(np.degrees(max(diff[:6]))),float(diff[6])],comparisons=results)


if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1]),json.loads(sys.argv[2])),allow_nan=False))
