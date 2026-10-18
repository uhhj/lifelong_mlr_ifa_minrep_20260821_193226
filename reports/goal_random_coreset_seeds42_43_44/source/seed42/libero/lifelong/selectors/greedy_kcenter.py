"""Frozen exploratory distance; no global RNG or evaluation information."""
import random
import time
from pathlib import Path
import numpy as np

SETTINGS = dict(version='visual_action_kcenter_v1', vision_tokens=[1, 2],
    time_positions=8, visual_weight=0.5, action_weight=0.5, calibration_pairs=4096,
    reference_count=256, seed=42, block_size=256, numeric_dtype='float64',
    action_scale='MAD*1.4826; then IQR/1.349; then range; then 1',
    action_distance='RMS standardized last-step action difference',
    distance_scale='median strictly positive calibration distances, or 1 if all zero',
    center='minimum mean combined distance to seeded reference subset',
    tie_break='lexicographic filename; absolute float64 tie tolerance 1e-12', clipping=False)

def robust_scale(actions):
    a=np.asarray(actions,dtype=np.float64)
    med=np.median(a,axis=0)
    candidates=[np.median(np.abs(a-med),axis=0)*1.4826,
                (np.percentile(a,75,axis=0)-np.percentile(a,25,axis=0))/1.349,
                np.ptp(a,axis=0),np.ones(a.shape[1])]
    out=candidates[0].copy()
    for c in candidates[1:]:out=np.where(out>1e-12,out,c)
    return out

def reference_pairs(n,seed=42,limit=4096):
    total=n*(n-1)//2
    flat=sorted(random.Random(seed).sample(range(total),min(limit,total)))
    rows=[];i=0;offset=0
    for q in flat:
        while q>=offset+n-i-1:offset+=n-i-1;i+=1
        rows.append((i,i+1+q-offset))
    return np.asarray(rows,dtype=np.int64).reshape(-1,2)

class Distance:
    def __init__(self,vision,actions,action_population,seed=42,block=256):
        v=np.asarray(vision,dtype=np.float64);a=np.asarray(actions,dtype=np.float64)
        assert v.ndim==4 and v.shape[1:3]==(8,2) and a.shape==(len(v),7)
        assert len(v)>0 and np.isfinite(v).all() and np.isfinite(a).all()
        norms=np.linalg.norm(v,axis=-1,keepdims=True)
        assert (norms>1e-12).all(),'Undefined cosine: zero visual token'
        self.v=(v/norms).reshape(len(v),-1);self.block=block
        self.action_scale=robust_scale(action_population)
        self.a=a/self.action_scale
        self.pairs=reference_pairs(len(v),seed)
        if len(self.pairs):
            i,j=self.pairs.T
            vh=np.maximum(0,1-np.sum(self.v[i]*self.v[j],axis=1)/16)
            da=np.sqrt(np.mean((self.a[i]-self.a[j])**2,axis=1))
        else:vh=da=np.zeros(0)
        self.hs=float(np.median(vh[vh>1e-12])) if np.any(vh>1e-12) else 1.
        self.as_=float(np.median(da[da>1e-12])) if np.any(da>1e-12) else 1.
    def distances(self,ids):
        ids=np.asarray(ids,dtype=np.int64);out=np.empty((len(self.v),len(ids)))
        for start in range(0,len(self.v),self.block):
            end=min(start+self.block,len(self.v))
            vh=np.maximum(0,1-self.v[start:end]@self.v[ids].T/16)
            da=np.sqrt(np.mean((self.a[start:end,None,:]-self.a[ids][None,:,:])**2,axis=-1))
            out[start:end]=.5*vh/self.hs+.5*da/self.as_
        out[ids,np.arange(len(ids))]=0.
        return out

def select_arrays(ids,vision,actions,action_population,k,seed=42,block=256):
    assert len(ids)==len(set(ids)) and 1<=k<=len(ids)
    order=np.argsort(np.asarray(ids));ids=[ids[i] for i in order]
    d=Distance(np.asarray(vision)[order],np.asarray(actions)[order],action_population,seed,block)
    refs=sorted(random.Random(seed).sample(range(len(ids)),min(256,len(ids))))
    centrality=d.distances(refs).mean(axis=1)
    center=int(np.flatnonzero(centrality<=centrality.min()+1e-12)[0])
    chosen=[center];nearest=d.distances([center])[:,0];radii=[float(nearest.max())]
    while len(chosen)<k:
        nearest[chosen]=-np.inf
        nxt=int(np.flatnonzero(nearest>=nearest.max()-1e-12)[0]);chosen.append(nxt)
        nearest=np.minimum(nearest,d.distances([nxt])[:,0])
        rest=np.ones(len(ids),dtype=bool);rest[chosen]=False
        radii.append(float(nearest[rest].max()) if rest.any() else 0.)
    return [ids[i] for i in chosen],dict(settings=SETTINGS,initial_center=ids[center],
        reference_ids=[ids[i] for i in refs],calibration_pair_ids=[[ids[i],ids[j]] for i,j in d.pairs],
        action_scale=d.action_scale.tolist(),visual_scale=d.hs,action_distance_scale=d.as_,
        covering_radius=radii)

def greedy_kcenter_select(paths,k,seed=42,action_population=None):
    import torch
    begin=time.monotonic();paths=sorted(paths,key=lambda p:Path(p).name)
    assert action_population is not None,'Raw current-task demonstration actions required'
    vv=[];aa=[]
    for path in paths:
        x=torch.load(path,map_location='cpu');v=x['feature'].detach().cpu().numpy();a=x['actions'].detach().cpu().numpy()
        assert v.shape==(1,8,4,512) and a.shape==(1,8,7)
        vv.append(v[0,:,1:3,:]);aa.append(a[0,-1,:])
    names,report=select_arrays([Path(p).name for p in paths],vv,aa,action_population,k,seed)
    lookup={Path(p).name:p for p in paths};report['seconds']=time.monotonic()-begin
    return [lookup[n] for n in names],report
