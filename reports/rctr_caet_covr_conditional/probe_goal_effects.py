"""Zero-update physical-label and cross-task-pair feasibility for conditional GOAL."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from caet_exp.labels import recorded_effects, TARGETS, certified
from caet_exp.transfer import pair_masks

torch.set_num_threads(1)
reports={}
for seed in (41,42,43):
    date="20261018" if seed==42 else "20261019"
    root=Path("/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_%s"%(seed,date))
    labels={};tasks={}
    for task in range(6,10):
        b=json.loads((root/"bindings"/("task%d.json"%task)).read_text())
        demos=sorted({r["demo_id"] for r in b["windows"].values()},
                     key=lambda s:int(s.split("_")[-1]))
        holdout=demos[-1];effects={};actions={};metadata={}
        with h5py.File(b["hdf5"]) as f:
            for demo in demos:
                effects[demo],metadata[demo]=recorded_effects(f["data"][demo],TARGETS[task])
                actions[demo]=f["data"][demo]["actions"][:].astype(np.float32)
        rows={}
        for name,row in b["windows"].items():
            assert certified(row)
            raw=actions[row["demo_id"]]
            payload=torch.load(root/"features"/str(task)/name,map_location="cpu")["actions"][0].numpy()
            assert payload.shape==(8,7)
            assert len(raw)==row["length"]
            assert all(np.array_equal(payload,raw[np.minimum(np.arange(start,start+8),len(raw)-1)])
                       for start in row["possible_starts"])
            target=row["target"]
            if target<len(effects[row["demo_id"]]):
                e=effects[row["demo_id"]][target].astype(np.float32)
                rows[name]=dict(effect=torch.from_numpy(e),task=task,holdout=row["demo_id"]==holdout,
                    moving=bool(np.linalg.norm(e[:3])>=.001 or np.linalg.norm(e[3:])>=np.pi/180))
        labels[task]=rows
        tasks[task]=dict(windows=len(b["windows"]),valid=len(rows),invalid=len(b["windows"])-len(rows),
                        moving=sum(r["moving"] for r in rows.values()),target=TARGETS[task],
                        alignment_and_full_window_binding=True,physical_layout=metadata[demos[0]])
    stages={}
    for stage in (7,8,9):
        accessible=list(labels[stage].values())
        for task in range(6,stage):
            m=json.loads((root/"ra_coreset/storage"/("task%d.json"%task)).read_text())
            ids=[Path(r["source"]).name for r in m["selected"]]
            accessible.extend(labels[task][i] for i in ids if i in labels[task])
        train=[r for r in accessible if not r["holdout"]]
        effect=torch.stack([r["effect"] for r in train])
        center=effect.double().mean(0).float()
        scale=effect.double().std(0,unbiased=False).float().clamp_min(1e-6)
        positive,negative=pair_masks((effect-center)/scale,torch.tensor([r["moving"] for r in train]),
                                    torch.tensor([r["task"] for r in train]))
        stages[stage]=dict(training_valid=len(train),heldout_valid=len(accessible)-len(train),
                          positive_pairs=int(positive.sum()),negative_pairs=int(negative.sum()))
    reports[seed]=dict(tasks=tasks,stages=stages,
        pair_feasible=all(v["positive_pairs"]>0 and v["negative_pairs"]>0 for v in stages.values()))
print(json.dumps(dict(status="PASS",kind="READ_ONLY_FEASIBILITY_NOT_FITTED_OR_GPU",
    optimizer_updates=0,environment_steps=0,seeds=reports),indent=2))
