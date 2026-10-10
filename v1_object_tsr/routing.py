"""Source-only routing over the unchanged native epoch slots."""
import hashlib
import json
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler


def probabilities(p0,cosine,current_valid=True,epsilon=.5,tau=.1):
    if not p0:return {}
    if not current_valid or epsilon==1:return dict(p0)
    valid=[t for t in sorted(p0) if cosine.get(t) is not None and np.isfinite(cosine[t])]
    out=dict(p0)
    if not valid:return out
    mass=sum(p0[t] for t in valid)
    logits=np.array([cosine[t]/tau for t in valid],dtype=np.float64)
    q=np.array([p0[t] for t in valid])*np.exp(logits-logits.max());q/=q.sum()
    for t,v in zip(valid,q):out[t]=epsilon*p0[t]+(1-epsilon)*mass*float(v)
    assert abs(sum(out.values())-1)<1e-12
    return out


def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()


class RoutedSampler:
    def __init__(self,dataset,task,seed,prob,epsilon=.5,task_ids=None):
        self.native=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True)
        if task_ids is None:
            from libero.lifelong.algos.mlr import parse_path
            task_ids=[int(parse_path(p)) for p in dataset.file_paths]
        self.ids=list(task_ids);self.task=task;self.seed=seed;self.prob=prob;self.epsilon=epsilon;self.epoch=0
        assert all(6<=t<=task for t in self.ids),'FUTURE_OR_BASE_TASK_IN_REPLAY'
        self.groups={t:[i for i,s in enumerate(self.ids) if s==t] for t in sorted(prob)}
        assert all(self.groups.values())
    def __len__(self):return len(self.native)
    def set_epoch(self,epoch):self.epoch=epoch;self.native.set_epoch(epoch)
    def __iter__(self):
        original=list(iter(self.native));routed=list(original)
        # One source and invalid-current fallbacks must retain native order too.
        p0={t:len(g)/sum(map(len,self.groups.values())) for t,g in self.groups.items()}
        direct=self.epsilon==1 or len(self.prob)<2 or self.prob==p0
        if not direct:
            choose=random.Random('TSR-source:%d:%d:%d'%(self.seed,self.task,self.epoch))
            inside={t:random.Random('TSR-within:%d:%d:%d:%d'%(self.seed,self.task,self.epoch,t)) for t in self.groups}
            pools={t:[] for t in self.groups};tasks=sorted(self.prob)
            for pos,i in enumerate(original):
                if self.ids[i]==self.task:continue
                t=choose.choices(tasks,weights=[self.prob[t] for t in tasks],k=1)[0]
                if not pools[t]:pools[t]=self.groups[t].copy();inside[t].shuffle(pools[t])
                routed[pos]=pools[t].pop()
        assert all(a==b for a,b in zip(original,routed) if self.ids[a]==self.task)
        assert all((self.ids[a]==self.task)==(self.ids[b]==self.task) for a,b in zip(original,routed))
        self.last=dict(epoch=self.epoch,task=self.task,direct_native=direct,total=len(routed),
            current_count=sum(self.ids[i]==self.task for i in routed),old_count=sum(self.ids[i]<self.task for i in routed),
            original_index_sha256=digest(original),routed_index_sha256=digest(routed),
            current_index_sha256=digest([i for i in routed if self.ids[i]==self.task]),
            selected_counts={str(t):sum(self.ids[i]==t for i in routed) for t in sorted(set(self.ids))})
        return iter(routed)
