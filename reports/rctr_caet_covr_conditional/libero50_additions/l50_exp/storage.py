"""Same per-task RA selection; immutable new buffer hardlinks save physical disk."""
import json
import math
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from libero.lifelong.selectors.greedy_kcenter import greedy_kcenter_select,Distance
from ra_exp.selector_reference import select_anchor_repair
from .common import O,SEED,sha,write,rng,restore_rng
from .data import cfg,tasks
from .selection_verify import independent_verify

def buffer_hashes():
    return {p.name:sha(p) for p in sorted((O/"ra_coreset/buffer").glob("*.pt"))}

def selection(task):
    paths=sorted((O/"features"/str(task)).glob("*.pt"))
    assert paths
    k=math.ceil(len(paths)*.5);state=rng();began=time.monotonic()
    try:
        c=cfg("train");r=tasks()[task]
        feature=json.loads((O/"FEATURES.json").read_text())["tasks"][str(task)]
        with h5py.File(Path(c.folder)/r["dataset_relative_path"]) as f:
            population=np.concatenate([f["data"][d]["actions"][:] for d in feature["demos"]])
        s0,original=greedy_kcenter_select([str(p) for p in paths],k,42,population)
        payload=[torch.load(p,map_location="cpu") for p in paths]
        v=np.array([x["feature"][0,:,1:3,:].numpy() for x in payload])
        a=np.array([x["actions"][0,-1,:].numpy() for x in payload])
        distance=Distance(v,a,population,42);D=distance.distances(np.arange(len(paths)))
        assert distance.hs==original["visual_scale"] and distance.as_==original["action_distance_scale"]
        ids=[p.name for p in paths]
        selected,report=select_anchor_repair(ids,[Path(p).name for p in s0],D,seed=SEED)
        assert len(set(selected))==k
        independent_verify(ids,report,D)
        report.update(original_selection=original,original_ids=[Path(p).name for p in s0],
                      selected_ids=selected,seconds=time.monotonic()-began,
                      independent_all_deletion_choices=True)
        return selected,report
    finally:restore_rng(state)

def store(task):
    path=O/"ra_coreset/storage"/("task%d.json"%task)
    buffer=O/"ra_coreset/buffer";buffer.mkdir(parents=True,exist_ok=True)
    if path.exists():
        m=json.loads(path.read_text());assert m["contract_sha256"]==sha(O/"CONTRACT.json")
    else:
        ids,report=selection(task)
        paths=O/"features"/str(task)
        m=dict(task=task,K=len(ids),before=buffer_hashes(),selection=report,
            selected=[dict(source=str(paths/n),destination="t%d__%s"%(task,n),sha256=sha(paths/n)) for n in ids],
            contract_sha256=sha(O/"CONTRACT.json"),storage="hardlink immutable own new feature; original payload bytes")
        write(path,m) # Committed selection precedes copying/linking.
    allowed=dict(m["before"]);allowed.update({r["destination"]:r["sha256"] for r in m["selected"]})
    actual=buffer_hashes();assert set(actual)<=set(allowed)
    assert all(allowed[n]==h for n,h in actual.items())
    for row in m["selected"]:
        src=Path(row["source"]);dst=buffer/row["destination"]
        assert sha(src)==row["sha256"]
        if dst.exists():assert sha(dst)==row["sha256"]
        else:
            assert src.stat().st_dev==buffer.stat().st_dev
            os.link(src,dst)
        assert sha(dst)==row["sha256"]
    assert buffer_hashes()==allowed
    write(O/"ra_coreset"/("buffer_after_task%d.json"%task),
          dict(task=task,total=len(allowed),hashes=allowed,source_payload_unchanged=True))
