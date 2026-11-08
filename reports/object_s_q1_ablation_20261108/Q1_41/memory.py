"""Immutable acquisition-time caches; uniform old-task slots, independent sampling RNG."""
from runtime import *
from losses import retain

def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def load_memory(arm,stage):
    out={}
    for task in range(6,stage):
        p=memory_path(arm,task);m=js(R/('MEMORY_%s_%d.json'%(arm,task)))
        assert sha(p)==m['cache_sha256'] and m['count']<=MEMORY_PER_TASK
        x=torch.load(p,map_location='cpu');assert len(x['rows'])==m['count']
        assert x['source_sha256']==m['source_sha256']==sha(model_path(arm,task))
        for row,meta in zip(x['rows'],m['rows']):
            assert tensor_sha(row['feature'])==meta['feature_sha256']
            assert tensor_sha(row['mu'])==meta['mu_sha256'] and tensor_sha(row['sd'])==meta['sd_sha256']
            assert 1<=len(row['feature'])<=8 and tuple(row['feature'].shape[1:])==(4,512)
        out[task]=x
    assert sum(len(x['rows']) for x in out.values())<=512
    return out

def slots(memory,stage,step):
    tasks=list(range(6,stage));out=[]
    if not tasks:return out
    for slot in range(ANCHOR_BATCH):
        task=tasks[(step*ANCHOR_BATCH+slot)%len(tasks)];rows=memory[task]['rows']
        if not rows:out.append((task,None));continue
        g=np.random.RandomState(20261025+stage*1000000+step*ANCHOR_BATCH+slot)
        out.append((task,int(g.randint(len(rows)))))
    return out

def retention(policy,memory,stage,step):
    selection=slots(memory,stage,step);zero=next(p for p in policy.parameters() if p.requires_grad).sum()*0
    ex=zero;rt=zero;forwards=0;actual=0;groups={}
    for task,i in selection:
        if i is not None:
            row=memory[task]['rows'][i];groups.setdefault((task,len(row['feature'])),[]).append(row)
    for (task,length),rows in sorted(groups.items()):
        x=torch.stack([r['feature'] for r in rows]).cuda()
        tm=torch.stack([r['mu'] for r in rows]).cuda();ts=torch.stack([r['sd'] for r in rows]).cuda()
        assert x.ndim==4 and 1<=x.shape[1]<=8 and tuple(x.shape[2:])==(4,512)
        q=policy.policy_head.forward_fn(policy.temporal_encode(x)[:,-1])
        a,b=retain(q,tm,ts,memory[task]['scales'].cuda())
        ex=ex+a.sum()/ANCHOR_BATCH;rt=rt+b.sum()/ANCHOR_BATCH
        forwards+=1;actual+=len(rows)
    return ex,rt,dict(slots=len(selection),actual=actual,empty_slots=len(selection)-actual,forward_calls=forwards,selection=selection)
