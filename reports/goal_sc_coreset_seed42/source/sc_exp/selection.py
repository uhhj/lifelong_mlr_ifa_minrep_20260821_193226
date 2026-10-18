"""SC-Coreset: exact proportional joint quotas and minimum-count geometric swaps."""
import itertools,time
from collections import Counter
import numpy as np

def group_id(target,length,grip):
    assert 0<=target<length and np.isfinite(grip)
    # Integer arithmetic avoids boundary rounding; length=1 is a true endpoint.
    time_bin=4 if target==length-1 else min(3,4*target//max(1,length-1))
    sign=0 if grip<0 else (1 if grip==0 else 2)
    return time_bin*3+sign

def quotas(groups,selected):
    n=len(groups);q=len(selected);counts=Counter(groups);old=Counter(groups[i] for i in selected)
    gids=sorted(counts);lo={g:q*counts[g]//n for g in gids};hi={g:(q*counts[g]+n-1)//n for g in gids}
    flexible=[g for g in gids if lo[g]!=hi[g]];needed=q-sum(lo.values());best=None
    for upper in itertools.combinations(flexible,needed):
        z=dict(lo)
        for g in upper:z[g]+=1
        l1=sum(abs(old[g]-z[g]) for g in gids)
        # All denominators are n, so this is exact total absolute rounding error.
        err=sum(abs(z[g]*n-q*counts[g]) for g in gids)
        key=(l1,err,tuple(-z[g] for g in gids))
        if best is None or key<best[0]:best=(key,z)
    assert best is not None and best[0][0]%2==0
    return best[1],dict(population=dict(counts),original=dict(old),lower=lo,upper=hi,
                        minimum_swaps=best[0][0]//2,rounding_error_numerator=best[0][1],
                        rounding_tie='min L1; min total absolute ideal error; earlier group ID receives ceil')

def repair(ids,groups,original,D):
    assert ids==sorted(ids) and len(ids)==len(set(ids)) and len(groups)==len(ids)
    lookup={n:i for i,n in enumerate(ids)};chosen=[lookup[n] for n in original]
    assert len(chosen)==len(set(chosen)) and D.shape==(len(ids),len(ids)) and np.isfinite(D).all()
    q,report=quotas(groups,chosen);counts=Counter(groups[i] for i in chosen);original_set=set(chosen)
    before=D[:,chosen].min(1);swaps=[]
    for step in range(report['minimum_swaps']):
        nearest=D[:,chosen].min(1)
        eligible=[i for i in range(len(ids)) if i not in set(chosen) and counts[groups[i]]<q[groups[i]]]
        far=max(nearest[i] for i in eligible)
        x=next(i for i in eligible if nearest[i]>=far-1e-12)
        augmented=chosen+[x];dist=D[:,augmented]
        # Exact leave-one-center-out radius from nearest/second-nearest distances.
        order=np.argsort(dist,axis=1,kind='stable')[:,:2]
        first=dist[np.arange(len(ids)),order[:,0]];second=dist[np.arange(len(ids)),order[:,1]]
        deletion=np.full(len(augmented),float(first.max()))
        np.maximum.at(deletion,order[:,0],second)
        candidates=sorted((i for i in chosen if counts[groups[i]]>q[groups[i]]))
        costs={i:float(deletion[augmented.index(i)]) for i in candidates}
        best=min(costs.values());y=next(i for i in candidates if costs[i]<=best+1e-12)
        slot=chosen.index(y);chosen[slot]=x;counts[groups[y]]-=1;counts[groups[x]]+=1
        swaps.append(dict(step=step,added=ids[x],removed=ids[y],slot=slot,
                          added_nearest_distance=float(nearest[x]),radius_after=costs[y]))
    assert all(counts[g]==q[g] for g in q) and len(chosen)==len(set(chosen))
    assert len(original_set-set(chosen))==report['minimum_swaps']
    for g in q:
        if report['original'].get(g,0)<=q[g]:assert all(i in chosen for i in original_set if groups[i]==g)
        assert abs(counts[g]*len(ids)-len(chosen)*report['population'][g])<len(ids)
    after=D[:,chosen].min(1)
    report.update(target=q,final=dict(counts),swaps=swaps,identity_noop=not swaps,
                  retained_original=len(original_set&set(chosen)),
                  radius_before=float(before.max()),radius_after=float(after.max()),
                  mean_distance_before=float(before.mean()),mean_distance_after=float(after.mean()),
                  p95_before=float(np.quantile(before,.95)),p95_after=float(np.quantile(after,.95)),
                  order_rule='replace removed member in its existing slot; otherwise preserve original order')
    return [ids[i] for i in chosen],report
