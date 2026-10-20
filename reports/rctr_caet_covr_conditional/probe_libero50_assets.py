"""Read only acquired datasets and existing environment assets, zero policy calls."""
import hashlib
import json
import math
from pathlib import Path
import h5py
import torch

Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
P=json.loads((Q/"LIBERO50_PARTITION.json").read_text())
T=json.loads((Q/"LIBERO50_PARENT_EVAL_TEMPLATE.json").read_text())
DATA=Path("/data/mlr_ifa/datasets/libero50_acquired")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()
rows=[]
all_rows=P["pretraining"]+sum(P["lifelong_stages"],[])
for i,row in enumerate(all_rows):
    b=Path(T["bddl_folder"])/row["bddl_relative_path"]
    init=Path(T["init_states_folder"])/row["init_relative_path"]
    assert b.is_file() and init.is_file(),(str(b),str(init))
    states=torch.load(init,map_location="cpu")
    assert len(states)>=20
    r=dict(name=row["name"],pretraining=i<25,bddl_sha256=sha(b),initial_sha256=sha(init),
           initial_count=len(states),dataset_verified=False)
    path=DATA/row["dataset_relative_path"]
    receipt=Q/"libero50_acquisition"/(Path(row["dataset_relative_path"]).stem+".json")
    if receipt.exists():
        proof=json.loads(receipt.read_text())
        assert proof["upstream_sha256"]==row["raw_sha256"]
        assert proof["all_dataset_bytes_equal"] and proof["all_attributes_equal"]
        assert sha(path)==proof["packed_sha256"]
        with h5py.File(path) as f:
            names=sorted(f["data"],key=lambda x:int(x.split("_")[-1]))
            assert len(names)==50
            lengths=[len(f["data"][d]["actions"]) for d in names]
            for d in names:
                g=f["data"][d]
                assert g["actions"].shape[1:]==(7,)
                assert g["obs/agentview_rgb"].dtype.name=="uint8"
                assert g["obs/eye_in_hand_rgb"].dtype.name=="uint8"
                assert all(len(v)==len(g["actions"]) for v in g["obs"].values())
            r.update(dataset_verified=True,all_demo_lengths=lengths,
                demos=names,full_windows=sum(lengths),lifelong_windows=sum(lengths[:10]),
                packed_sha256=proof["packed_sha256"],logical_array_sha256=proof["logical_array_sha256"])
    rows.append(r)
ready=all(x["dataset_verified"] for x in rows)
out=dict(status="DATA_AND_ENV_METADATA_READY" if ready else "PARTIAL_ACQUISITION",
    environment_steps=0,optimizer_calls=0,rows=rows,verified=sum(x["dataset_verified"] for x in rows),
    official_task_pretrain_loaded=False)
if ready:
    count={x["name"]:x["lifelong_windows"] for x in rows}
    old=0;stages=[]
    for stage,tasks in enumerate(P["lifelong_stages"],1):
        current=sum(count[t["name"]] for t in tasks)
        stages.append(dict(stage=stage,current_windows=current,old_buffer_windows=old,
                           batches_per_epoch=math.ceil((current+old)/10)))
        old+=sum(math.ceil(count[t["name"]]/2) for t in tasks)
    out.update(stage_counts=stages,main_steps_per_seed=100*sum(x["batches_per_epoch"] for x in stages),
        shared_pretrain_steps=100*math.ceil(sum(x["full_windows"] for x in rows[:25])/10))
(Q/"LIBERO50_ASSET_PROBE.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="rows"}))
