"""Use native window construction, padding, tokenization and observation normalization."""
import json
from pathlib import Path
from easydict import EasyDict
from torch.utils.data import Dataset
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import get_dataset,SequenceVLDataset,GroupedTaskDataset
from libero.lifelong.utils import get_task_embs
from .common import W,O

def partition():return json.loads((W/"PARTITION.json").read_text())
def tasks():return sum(partition()["lifelong_stages"],[])
def cfg(kind):
    c=EasyDict(json.loads((O/("resolved_"+kind+".json")).read_text()))
    assert c.gamma==.1 and c.policy.use_state and c.policy.use_language and c.policy.use_vision
    return c
def native_task(row):
    # Resolve the frozen full partition directly, without the legacy filtered
    # benchmark constructor that omits six SCENE9 kitchen tasks.
    from libero.libero.benchmark import task_maps
    t=task_maps[row["source_suite"]][row["name"]]
    assert t.problem_folder+"/"+t.bddl_file==row["bddl_relative_path"]
    assert t.problem_folder+"/"+t.init_states_file==row["init_relative_path"]
    assert t.problem_folder+"/"+t.name+"_demo.hdf5"==row["dataset_relative_path"]
    return t
def datasets(c,rows,demo_limit):
    raw=[];shape=None;descriptions=[]
    for i,row in enumerate(rows):
        d,meta=get_dataset(str(Path(c.folder)/row["dataset_relative_path"]),
            c.data.obs.modality,initialize_obs_utils=i==0,seq_len=8,demo_limit=demo_limit,train=c.training)
        assert d.n_demos==demo_limit
        if shape is not None:assert meta==shape
        shape=meta;raw.append(d);descriptions.append(native_task(row).language)
    emb=get_task_embs(c,descriptions)
    return raw,emb,shape
def pretrain_dataset(c):
    rows=partition()["pretraining"];raw,emb,shape=datasets(c,rows,50)
    return GroupedTaskDataset(raw,emb["input_ids"],emb["attention_mask"]),shape
def lifelong_datasets(c):
    raw,emb,shape=datasets(c,tasks(),10)
    return [SequenceVLDataset(d,emb["input_ids"][i],emb["attention_mask"][i])
            for i,d in enumerate(raw)],emb,shape
class Indexed(Dataset):
    """Actual accessed native index; never iterates a second sampler."""
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):return self.dataset[i],i
