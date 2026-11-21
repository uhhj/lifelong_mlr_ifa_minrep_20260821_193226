"""Evaluate the deployed B1 action head on archived held-out visual/state histories."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch

work=Path(sys.argv[1])
for p in Path("/proc").glob("[0-9]*/cmdline"):
    try:args=p.read_bytes().split(b"\0")
    except OSError:continue
    if any(a.endswith(b"/runner.py") and b in (b"infer",b"guard") for a,b in zip(args,args[1:])):
        raise RuntimeError("Live policy session on host")
desktop=Path(subprocess.check_output(["xdg-user-dir","DESKTOP"],text=True).strip())
exec_line=next(s[5:] for s in (desktop/"PiPER-Policy.desktop").read_text().splitlines() if s.startswith("Exec="))
sys.path.insert(0,str(Path(shlex.split(exec_line)[-1]).parent))
from policy import Predictor

torch.set_num_threads(2)
torch.set_grad_enabled(False)
predictor=Predictor(work,0)
from piper_base import GOAL_SHA, TOKEN_SHA, SETS
run=work/"offline_training/b123_goal_v1"
base=work/"expert_sets"/SETS[0][0]
manifest=(base/"manifest.json").read_bytes()
assert hashlib.sha256(manifest).hexdigest()==SETS[0][1]
split=json.loads((run/"split.json").read_bytes())
cache_key=hashlib.sha256((run/"piper_base.py").read_bytes()).hexdigest()+GOAL_SHA+TOKEN_SHA
datasets=[]
for ep in json.loads(manifest)["episodes"]:
    if split[ep["episode_id"]]!="val":continue
    content=(base/ep["episode_dir"]/"alignment.jsonl").read_bytes()
    assert hashlib.sha256(content).hexdigest()==ep["alignment_sha256"]
    rows=[json.loads(s) for s in content.splitlines()][ep["first_action_row"]:ep["last_action_row"]+1]
    raw=torch.load(run/"visual_cache"/(ep["episode_id"]+".pt"),map_location="cpu",weights_only=True)
    assert raw["identity"]==cache_key and len(raw["vision"])==len(rows)
    q,a=(torch.tensor([r[k]["joints_rad"]+[r[k]["gripper_mm"]] for r in rows],dtype=torch.float32).cuda()
         for k in ("feedback","target"))
    v=raw["vision"].cuda()
    output=[]
    for begin in range(0,len(rows),64):
        last=torch.arange(begin,min(begin+64,len(rows)),device="cuda")
        idx=(last[:,None]+torch.arange(-7,1,device="cuda")[None]).clamp_min(0)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            _,pred=predictor.model(v[idx],q[idx],torch.zeros_like(last))
        output.append(pred.float())
    pred=torch.cat(output)
    datasets.append(tuple(t.cpu().numpy() for t in (q,a,pred)))
assert len(datasets)==10
q,a,p=(np.concatenate(arrays) for arrays in zip(*datasets))
assert len(q)==3145 and np.isfinite(p).all()
target=np.column_stack([np.degrees(a[:,0]-q[:,0]),a[:,6]-q[:,6]])
predicted=np.column_stack([np.degrees(p[:,0]-q[:,0]),p[:,6]-q[:,6]])
metrics={}
for name,k,sgn,cut in (("j1_positive",0,1,.5),("j1_negative",0,-1,.5),
                       ("close_grip",1,-1,5),("open_grip",1,1,5)):
    valid=sgn*target[:,k]>=cut
    j=predicted[valid,k]
    metrics[name]=dict(selected_frames=int(valid.sum()),same_direction=int((j*sgn>0).sum()),
                       reaches_cut=int((j*sgn>=cut).sum()),median_expert=round(float(np.median(target[valid,k])),3),
                       median_predicted=round(float(np.median(j)),3))
print(json.dumps(dict(episodes=len(datasets),frames=len(q),j1_cut_deg=.5,grip_cut_mm=5,
                      joint_mae_deg=round(float(np.abs(np.degrees(p[:,:6]-a[:,:6])).mean()),4),
                      gripper_mae_mm=round(float(np.abs(p[:,6]-a[:,6]).mean()),4),classes=metrics)))
