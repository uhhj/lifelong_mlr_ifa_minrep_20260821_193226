"""CPU-only read of existing logs/RNG; no models, CUDA, training or environments."""
import json, hashlib, pickle
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

roots = {
    "RA": Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019/ra_coreset"),
    "v2": Path("/data/mlr_ifa/experiments/Conditional_ZRv2_OBJECT_RA_seed41_20261019"),
    "v3": Path("/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_v3"),
}
hashes = {}
def read(p):
    raw = p.read_bytes()
    hashes[str(p)] = hashlib.sha256(raw).hexdigest()
    return raw
def digest(v):
    if torch.is_tensor(v):
        a=v.detach().cpu().contiguous()
        raw=str((str(a.dtype),tuple(a.shape))).encode()+a.numpy().tobytes()
    elif isinstance(v,np.ndarray):
        raw=str((str(v.dtype),v.shape)).encode()+v.tobytes()
    elif isinstance(v,dict):
        raw=json.dumps([(str(k),digest(v[k])) for k in sorted(v)]).encode()
    elif isinstance(v,(tuple,list)):
        raw=json.dumps([digest(x) for x in v]).encode()
    else:
        raw=pickle.dumps(v,protocol=4)
    return hashlib.sha256(raw).hexdigest()

rng=[]
for s in range(6,10):
    for t in range(6,s+1):
        row=dict(stage=s,task=t,groups={})
        for name,root in roots.items():
            p=root/"evaluation"/("rng_before_stage%d_task%d.pth"%(s,t))
            read(p)
            state=torch.load(str(p),map_location="cpu")
            row["groups"][name]={str(k):digest(v) for k,v in state.items()}
        row["equal_all"] = len({json.dumps(v,sort_keys=True) for v in row["groups"].values()})==1
        rng.append(row)
logs={}
for name in ("v2","v3"):
    root=roots[name]
    data={}
    for filename in ("STRUCTURE.jsonl","epochs.json","resolved_calibration.json"):
        raw=read(root/filename).decode()
        data[filename]=[json.loads(l) for l in raw.splitlines()] if filename.endswith("jsonl") else json.loads(raw)
    batches=[json.loads(l) for l in read(root/"calibration_batches.jsonl").decode().splitlines()]
    agg=defaultdict(list)
    for row in batches:agg[row["stage"],row["epoch"]].append(row)
    data["batch_summary"]=[]
    for (stage,epoch),rs in sorted(agg.items()):
        row=dict(stage=stage,epoch=epoch,batches=len(rs))
        for key in ("loss","grad_norm","delta_max","actual_kl_max","raw_kl_max","rounding_passes"):
            if key in rs[0]:
                vals=[r[key] for r in rs]
                row[key]=dict(min=min(vals),max=max(vals),mean=sum(vals)/len(vals))
        if "states" in rs[0]:
            row.update(states=sum(r["states"] for r in rs),projected_states=sum(r["projected_states"] for r in rs))
        row["task6_present_batches"]=sum(6 in r["active"] for r in rs)
        data["batch_summary"].append(row)
    data["batch_count"]=len(batches)
    logs[name]=data
configs={}
for name,root in roots.items():
    paths=set((root/"evaluation").glob("*config*.json")) | set((root/"evaluation").glob("resolved*.json"))
    configs[name]={p.name:json.loads(read(p)) for p in sorted(paths)}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status="PASS_READONLY",input_hashes=hashes,input_hashes_after_equal=True,
                     rng=rng,logs=logs,configs=configs,cuda_initialized=False,
                     new_model_forwards=0,new_optimizer_calls=0,new_environment_steps=0)))
