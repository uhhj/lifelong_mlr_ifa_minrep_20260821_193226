"""SC-v2 offline selector. Exact integer E; exhaustive legal one-swap search.

No model, optimizer, environment or RNG calls. Coverage ties use the parent's
absolute 1e-12 tolerance; E comparisons use integer numerators without tolerance.
"""
from collections import Counter
import time
import numpy as np

TOL = 1e-12


def prepare(ids, rows):
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    uids, groups, demos, endpoints = [], [], [], []
    identities = {}
    for name in ids:
        r = rows[name]
        t, length = r['target'], r['length']
        assert 0 <= t < length
        sign = r['group'] % 3  # Verified against actual last action by remote reader.
        quartile = min(3, 4*t // max(1, length-1)) if length > 1 else 3
        uid = (r['demo_id'], t)
        group = quartile*3 + sign
        signature = (group, length)
        assert uid not in identities or identities[uid] == signature
        identities[uid] = signature
        uids.append(uid); groups.append(group); demos.append(r['demo_id'])
        endpoints.append(t == length-1)
    ref = Counter(v[0] for v in identities.values())
    return dict(uids=uids, groups=np.asarray(groups), demos=demos,
                endpoints=endpoints, reference=ref, U=len(identities))


def error_num(chosen, meta):
    n = Counter(int(meta['groups'][i]) for i in chosen)
    return sum(abs(n[g]*meta['U']-len(chosen)*v) for g,v in meta['reference'].items())


def describe(chosen, meta, D):
    near = D[:, chosen].min(axis=1)
    return dict(E_numerator=error_num(chosen,meta), E_denominator=2*len(chosen)*meta['U'],
                E=error_num(chosen,meta)/(2*len(chosen)*meta['U']),
                radius=float(near.max()), mean=float(near.mean()),
                unique_times=len({meta['uids'][i] for i in chosen}),
                group_counts=dict(Counter(int(meta['groups'][i]) for i in chosen)),
                demo_counts=dict(Counter(meta['demos'][i] for i in chosen)),
                endpoint_demos=sorted({meta['demos'][i] for i in chosen if meta['endpoints'][i]}),
                endpoint_windows=sum(meta['endpoints'][i] for i in chosen))


def candidates(chosen, meta, protected_demo, protected_end):
    g = meta['groups']; U = meta['U']; q = len(chosen)
    counts = Counter(int(g[i]) for i in chosen)
    deficit = {k for k,v in meta['reference'].items() if counts[k]*U < q*v}
    surplus = {k for k,v in meta['reference'].items() if counts[k]*U > q*v}
    present = {meta['uids'][i] for i in chosen}
    xs = [i for i in range(len(g)) if g[i] in deficit and meta['uids'][i] not in present]
    ys = sorted(i for i in chosen if g[i] in surplus)
    base_error = error_num(chosen,meta)
    changes = {}
    for a in deficit:
        for b in surplus:
            av=counts[a]*U-q*meta['reference'][a]; bv=counts[b]*U-q*meta['reference'][b]
            changes[a,b] = base_error-abs(av)-abs(bv)+abs(av+U)+abs(bv-U)
    dcounts=Counter(meta['demos'][i] for i in chosen)
    ecounts=Counter(meta['demos'][i] for i in chosen if meta['endpoints'][i])
    valid=np.ones((len(ys),len(xs)),dtype=bool)
    errs=np.empty(valid.shape,dtype=np.int64)
    for j,y in enumerate(ys):
        for k,x in enumerate(xs):
            errs[j,k]=changes[int(g[x]),int(g[y])]
            valid[j,k] = errs[j,k]<base_error
            if meta['demos'][y] in protected_demo and dcounts[meta['demos'][y]] == 1:
                valid[j,k] &= meta['demos'][x] == meta['demos'][y]
            if meta['endpoints'][y] and meta['demos'][y] in protected_end and ecounts[meta['demos'][y]] == 1:
                # x cannot have the same endpoint UID, because it is already represented.
                valid[j,k] = False
    return xs,ys,valid,errs


def choose_pair(xs,ys,valid,errs,radii,means):
    if not valid.any(): return None
    e=int(errs[valid].min()); mask=valid & (errs==e)
    r=float(radii[mask].min()); mask &= radii<=r+TOL
    m=float(means[mask].min()); mask &= means<=m+TOL
    pairs=[(xs[k],ys[j],j,k) for j,k in zip(*np.nonzero(mask))]
    x,y,j,k=min(pairs)  # ids are lexicographically sorted: added ID, then removed ID.
    return x,y,dict(E_numerator=e,radius=float(radii[j,k]),mean=float(means[j,k]))


def search(chosen,meta,D,R0,M0,protected_demo,protected_end):
    xs,ys,valid,errs=candidates(chosen,meta,protected_demo,protected_end)
    info=dict(unseen_time_add_candidates=len(xs),surplus_remove_candidates=len(ys),
              pairs_total=len(xs)*len(ys),support_and_E_legal=int(valid.sum()))
    if not valid.any():
        info.update(coverage_legal=0); return None,info
    # For each candidate x, compute all deletion costs simultaneously. If y owns
    # a point's nearest center, removing y replaces its distance by min(second,Dx).
    # All other points keep min(first,Dx). Thus all pairs are evaluated exactly.
    dist=D[:,chosen]; owner=np.argmin(dist,axis=1)
    first=dist[np.arange(len(D)),owner]
    second=np.partition(dist,1,axis=1)[:,1] if len(chosen)>1 else np.full(len(D),np.inf)
    order=np.argsort(owner,kind='stable'); labels,starts=np.unique(owner[order],return_index=True)
    z=np.minimum(first[:,None],D[:,xs]); alt=np.minimum(second[:,None],D[:,xs])
    base_max=z.max(axis=0); base_sum=z.sum(axis=0)
    all_max=np.broadcast_to(base_max,(len(chosen),len(xs))).copy()
    all_sum=np.broadcast_to(base_sum,(len(chosen),len(xs))).copy()
    all_max[labels]=np.maximum(base_max,np.maximum.reduceat(alt[order],starts,axis=0))
    all_sum[labels]+=np.add.reduceat((alt-z)[order],starts,axis=0)
    slots=[chosen.index(y) for y in ys]
    radii=all_max[slots]; means=all_sum[slots]/len(D)
    valid &= (radii<=R0+TOL) & (means<=M0+TOL)
    info['coverage_legal']=int(valid.sum())
    return choose_pair(xs,ys,valid,errs,radii,means),info


def brute_search(chosen,meta,D,R0,M0,pdemo,pend):
    """Independent direct-nearest calculation for small synthetic verification."""
    xs,ys,valid,errs=candidates(chosen,meta,pdemo,pend)
    radii=np.zeros(valid.shape);means=np.zeros(valid.shape)
    for j,y in enumerate(ys):
        for k,x in enumerate(xs):
            if valid[j,k]:
                seq=[x if s==y else s for s in chosen]
                d=D[:,seq].min(axis=1);radii[j,k]=d.max();means[j,k]=d.mean()
                valid[j,k] &= radii[j,k]<=R0+TOL and means[j,k]<=M0+TOL
    return choose_pair(xs,ys,valid,errs,radii,means)


def repair(ids,rows,original,D,progress=None,verify_brute=False):
    begin=time.monotonic();meta=prepare(ids,rows);lookup={n:i for i,n in enumerate(ids)}
    chosen=[lookup[n] for n in original]
    assert len(chosen)==len(set(chosen)) and D.shape==(len(ids),len(ids)) and np.isfinite(D).all()
    before=describe(chosen,meta,D);R0=before['radius'];M0=before['mean']
    pdemo=set(before['demo_counts']);pend=set(before['endpoint_demos']);swaps=[]
    while True:
        best,info=search(chosen,meta,D,R0,M0,pdemo,pend)
        if verify_brute:
            brute=brute_search(chosen,meta,D,R0,M0,pdemo,pend)
            assert (None if best is None else best[:2]) == (None if brute is None else brute[:2])
        if best is None: break
        x,y,cost=best;slot=chosen.index(y);old=describe(chosen,meta,D)
        assert meta['uids'][x] not in {meta['uids'][i] for i in chosen}
        chosen[slot]=x;new=describe(chosen,meta,D)
        assert new['E_numerator']<old['E_numerator'] and new['E_numerator']==cost['E_numerator']
        assert abs(new['radius']-cost['radius'])<TOL and abs(new['mean']-cost['mean'])<TOL
        assert new['unique_times']>=old['unique_times']
        assert pdemo<=set(new['demo_counts']) and pend<=set(new['endpoint_demos'])
        assert new['radius']<=R0+TOL and new['mean']<=M0+TOL
        swaps.append(dict(step=len(swaps),added=ids[x],removed=ids[y],slot=slot,
                          after=new,search=info))
        if progress:progress(len(swaps),new)
    result=dict(original_ids=original,selected_ids=[ids[i] for i in chosen],
                reference_unique_counts=dict(meta['reference']),reference_unique_total=meta['U'],
                before=before,after=describe(chosen,meta,D),swaps=swaps,
                terminal_search=info,stop_reason='exhausted all legal one-swap improvements',
                no_intervention=not swaps,retained_original=len(set(original)&{ids[i] for i in chosen}),
                seconds=time.monotonic()-begin,tolerance=TOL,
                order_rule='replace removed slot, preserve all other positions',
                tie_rule='exact E numerator; radius within 1e-12; mean within 1e-12; added then removed lexicographic ID')
    if not swaps:assert result['selected_ids']==original
    return result
