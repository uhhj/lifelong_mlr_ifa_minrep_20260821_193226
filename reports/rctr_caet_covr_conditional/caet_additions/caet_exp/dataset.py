import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from coreset_exp.common import O, sha


class EffectDataset(Dataset):
    """Attach labels to actual native indices; never run another sampler."""
    def __init__(self, native, task):
        self.native = native
        self.file_paths = native.file_paths
        self.labels = {}
        current = O/"effects/current"/("task%d" % task)
        audit = json.loads((current/"AUDIT.json").read_text())
        assert sha(current/"labels.pth") == audit["labels_sha256"]
        labels = torch.load(current/"labels.pth", map_location="cpu")
        self.labels.update({str(O/"features"/str(task)/name):value for name,value in labels.items()})
        for old in range(6,task):
            path = O/"effects/memory"/("task%d.pth" % old)
            audit = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == audit["sha256"]
            kept = torch.load(path, map_location="cpu")
            self.labels.update({str(O/"ra_coreset/buffer"/name):value for name,value in kept.items()})
        assert all(str(p) in self.labels for p in self.file_paths)
        # Once per stage, actual allowed payload membership—not old discarded files.
        assert all(sha(p) == self.labels[str(p)]["sample_sha256"] for p in self.file_paths)

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        sample, task = self.native[index]
        label = self.labels[str(self.file_paths[index])]
        sample["effect_valid"] = label["valid"]
        sample["effect_target"] = label["effect"] if label["valid"] else torch.zeros(6)
        sample["effect_moving"] = label["moving"]
        sample["effect_holdout"] = label["holdout"]
        # Invalid zeros are padding for collation ONLY; all consumers use valid mask.
        return sample, task
