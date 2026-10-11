import hashlib,json,random,math
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler


def planned_cache(feature_root,task,seed):
    files=sorted((Path(feature_root)/str(task)).glob('*.pt'))
    return random.Random(seed).sample(files,max(1,math.ceil(len(files)*.5)))


class NativeSampler(DistributedSampler):
    def __init__(self,dataset,task):
        super().__init__(dataset,num_replicas=1,rank=0,shuffle=True)
        from libero.lifelong.algos.mlr import parse_path
        self.ids=[int(parse_path(p)) for p in dataset.file_paths];self.task=task
    def __iter__(self):
        order=list(super().__iter__())
        def h(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
        current=[i for i in order if self.ids[i]==self.task]
        self.last=dict(task=self.task,epoch=self.epoch,total=len(order),current_count=len(current),
            old_count=len(order)-len(current),original_index_sha256=h(order),routed_index_sha256=h(order),
            current_index_sha256=h(current),direct_native=True)
        return iter(order)
