"""Independent physical-label recomputation and allowed-memory audit; read only."""
import json
import math
import random
from pathlib import Path
import h5py
import torch
from coreset_exp.common import O, SEED, sha
from .labels import recorded_effects, TARGETS


def js(p):
    return json.loads(p.read_text())


def audit():
    rows=[]
    for task in range(6,10):
        root=O/"effects/current"/("task%d"%task)
        a=js(root/"AUDIT.json")
        assert a["status"]=="PASS" and sha(root/"labels.pth")==a["labels_sha256"]
        labels=torch.load(root/"labels.pth",map_location="cpu")
        binding=js(O/"bindings"/("task%d.json"%task))
        values={}
        with h5py.File(binding["hdf5"]) as h:
            for demo in a["demos"]:
                values[demo],_=recorded_effects(h["data"][demo],TARGETS[task])
        for name,item in binding["windows"].items():
            label=labels[name]
            valid=item["identity_unique"] and item["target"]<len(values[item["demo_id"]])
            assert label["valid"]==valid
            if valid:
                expected=torch.tensor(values[item["demo_id"]][item["target"]],dtype=torch.float32)
                assert torch.equal(label["effect"],expected)
            else:
                assert label["effect"] is None
        storage=js(O/"ra_coreset/storage"/("task%d.json"%task))
        kept=torch.load(O/"effects/memory"/("task%d.pth"%task),map_location="cpu")
        assert set(kept)=={r["destination"] for r in storage["selected"]}
        for r in storage["selected"]:
            assert kept[r["destination"]]["sample_sha256"]==r["sha256"]
        rows.append(dict(task=task,valid_labels=sum(v["valid"] for v in labels.values()),
                         retained=len(kept),physical_recomputation_exact=True))
    fits=[]
    for task in range(7,10):
        root=O/"effects/models"/("task%d"%task)
        a=js(root/"ACCEPTANCE.json")
        b=js(O/"ra_coreset/effect_bindings"/("task%d.json"%task))
        assert a["status"]=="PASS"
        assert sha(root/"final.pth")==a["frozen_predictor_sha256"]==b["model_sha256"]
        assert sha(root/"ACCEPTANCE.json")==b["acceptance_sha256"]
        payload=torch.load(root/"final.pth",map_location="cpu")
        allowed=[]
        for old in range(6,task):
            storage=js(O/"ra_coreset/storage"/("task%d.json"%old))
            allowed.extend(str(O/"ra_coreset/buffer"/r["destination"]) for r in storage["selected"])
        allowed=sorted(allowed)+[str(p) for p in sorted((O/"features"/str(task)).glob("*.pt"))]
        random.Random(SEED).shuffle(allowed)
        assert payload["input_paths"]==allowed, "Predictor read invalid memory population"
        epochs=js(root/"epochs.json")
        assert len(epochs)==10 and [r["epoch"] for r in epochs]==list(range(10))
        assert a["optimizer_steps"]==epochs[-1]["steps"]==10*math.ceil(a["train_windows"]/64)
        assert a["observed_training_positive_pairs"]>0
        assert a["action_gradient_finite"] and a["action_gradient_l1"]>0
        fits.append(a)
    return dict(labels=rows,predictors=fits)
