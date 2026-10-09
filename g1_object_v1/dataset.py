"""Original dataset order/content with explicit stable source identities."""
from pathlib import Path
from torch.utils.data import Dataset
from libero.lifelong.algos.mlr import build_dataset_for_task


class IdentifiedDataset(Dataset):
    def __init__(self,original):
        self.original=original
        self.file_paths=original.file_paths
    def __len__(self):return len(self.original)
    def __getitem__(self,i):
        sample,task=self.original[i]
        p=Path(self.file_paths[i])
        identity=p.name if p.name.startswith('t%d__'%int(task)) else 't%d__%s'%(int(task),p.name)
        return sample,task,identity


def identified_dataset(*args,**kwargs):return IdentifiedDataset(build_dataset_for_task(*args,**kwargs))
