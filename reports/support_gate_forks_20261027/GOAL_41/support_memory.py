from runtime import *
from legacy_memory import slots,tensor_sha
from losses import retain

def load_memory(arm,stage):
    out={}
    for task in range(6,stage):
        b=BINDING['caches'][str(task)];meta=b['metadata'];p=Path(b['path'])
        assert sha(p)==meta['cache_sha256'] and sha(b['teacher'])==meta['source_sha256']
        x=torch.load(p,map_location='cpu');assert len(x['rows'])==meta['count']
        ids=set()
        for row,m in zip(x['rows'],meta['rows']):
            assert tensor_sha(row['feature'])==m['feature_sha256'] and tensor_sha(row['mu'])==m['mu_sha256'] and tensor_sha(row['sd'])==m['sd_sha256']
            assert 1<=len(row['feature'])<=8 and row['source']=='online'
            ids.add(int(row['trajectory']))
        assert meta['training_attempts']==10 and len(ids)==meta['collection_successes']
        x['support']=len(ids)/10.;x['support_trajectories']=sorted(ids);out[task]=x
    return out

def weights(memory,arm):
    q={t:m['support'] for t,m in memory.items()}
    if arm=='S':return {t:1. for t in q}
    if arm=='U':return {t:sum(q.values())/len(q) for t in q}
    if arm in ['Q','G']:return q
    assert arm=='C';return {t:0. for t in q}

def from_selection(policy,memory,selection,ww,denominator):
    zero=next(p for p in policy.parameters() if p.requires_grad).sum()*0
    ex=zero;rt=zero;groups={};actual=0;contrib={}
    for task,i in selection:
        if i is not None:
            row=memory[task]['rows'][i];groups.setdefault((task,len(row['feature'])),[]).append(row)
    for (task,length),rows in sorted(groups.items()):
        x=torch.stack([r['feature'] for r in rows]).cuda();tm=torch.stack([r['mu'] for r in rows]).cuda();ts=torch.stack([r['sd'] for r in rows]).cuda()
        q=policy.policy_head.forward_fn(policy.temporal_encode(x)[:,-1]);a,b=retain(q,tm,ts,memory[task]['scales'].cuda())
        factor=ww[task]/denominator
        ea=a.sum()*factor;rb=b.sum()*factor;ex=ex+ea;rt=rt+rb;actual+=len(rows)
        v=contrib.setdefault(str(task),dict(windows=0,weighted_regression=0.,weighted_route=0.,support_weight=ww[task]))
        v['windows']+=len(rows);v['weighted_regression']+=float(ea.detach());v['weighted_route']+=float(rb.detach())
    return ex,rt,dict(slots=len(selection),actual=actual,empty_slots=len(selection)-actual,forward_calls=len(groups),selection=selection,contributions=contrib)

def retention(policy,memory,stage,step,arm='S'):
    return from_selection(policy,memory,slots(memory,stage,step),weights(memory,arm),ANCHOR_BATCH)
