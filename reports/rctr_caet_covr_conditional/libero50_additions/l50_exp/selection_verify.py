"""Independent full-candidate deletion oracle, inherited from verified RA admission."""
import numpy as np
def independent_verify(ids,report,D):
    lookup={n:i for i,n in enumerate(ids)};chosen=[lookup[n] for n in report['original_ids']];anchors=set(report['anchor_ids'])
    pending=[n for n in report['anchor_ids'] if n not in report['original_ids']]
    assert len(pending)==len(report['swaps'])
    for name,swap in zip(pending,report['swaps']):
        assert name==swap['incoming'];x=lookup[name];aug=chosen+[x]
        block=D[:,aug];owner=block.argmin(axis=1);ordered=np.sort(block,axis=1);first=ordered[:,0];second=ordered[:,1]
        winner=None
        for pos,old in sorted(enumerate(chosen),key=lambda z:ids[z[1]]):
            if ids[old] in anchors:continue
            nearest=np.where(owner==pos,second,first)
            r,m=float(nearest.max()),float(nearest.mean());oldname=ids[old]
            if winner is None or r<winner[0]-1e-12 or (abs(r-winner[0])<=1e-12 and m<winner[1]-1e-12) or (abs(r-winner[0])<=1e-12 and abs(m-winner[1])<=1e-12 and oldname<winner[2]):winner=(r,m,oldname,pos)
        assert winner[2]==swap['outgoing'] and winner[3]==swap['parent_center_slot']
        chosen[winner[3]]=x;v=D[:,chosen].min(axis=1)
        assert abs(float(v.max())-swap['radius_after'])<1e-12 and abs(float(v.mean())-swap['mean_after'])<1e-12
    assert [ids[i] for i in chosen]==report['selected_ids']
    assert len(set(report['selected_ids'])-set(report['original_ids']))==len(pending)
