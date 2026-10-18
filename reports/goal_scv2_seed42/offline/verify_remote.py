"""Independent real-distance swap search, using per-candidate reductions.
No selection implementation imported; no training or environment.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,hashlib,importlib.util,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch,h5py
torch.set_num_threads(2)
BASE=Path('/root/mlr_ifa/analyses/SCv2_GOAL_seed42_offline_20261018_verified')
SC=Path('/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018')
P=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017')
SRC=Path('/root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/source/libero/lifelong/selectors/greedy_kcenter.py')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def check_search(seq,D,ids,meta,ref,R0,M0,protected):
    U=sum(ref.values());q=len(seq)
    group=lambda i:min(3,4*meta[ids[i]]['target']//max(1,meta[ids[i]]['length']-1))*3+meta[ids[i]]['group']%3
    uid=lambda i:(meta[ids[i]]['demo_id'],meta[ids[i]]['target'])
    demo=lambda i:meta[ids[i]]['demo_id']
    endpoint=lambda i:meta[ids[i]]['target']==meta[ids[i]]['length']-1
    counts=Counter(group(i) for i in seq);present={uid(i) for i in seq}
    dc=Counter(demo(i) for i in seq);ec=Counter(demo(i) for i in seq if endpoint(i))
    base=sum(abs(counts[g]*U-q*n) for g,n in ref.items())
    a=D[:,seq];owner=a.argmin(1);first=a[np.arange(len(D)),owner];second=np.partition(a,1,axis=1)[:,1]
    totals=Counter();legal=[]
    for x in range(len(ids)):
        gx=group(x)
        if uid(x) in present or counts[gx]*U>=q*ref[gx]:continue
        z=np.minimum(first,D[:,x]);zz=np.minimum(second,D[:,x])
        correction=np.bincount(owner,weights=zz-z,minlength=q)
        max_by=np.full(q,float(z.max()));np.maximum.at(max_by,owner,zz)
        mean_by=(z.sum()+correction)/len(D)
        for slot,y in enumerate(seq):
            gy=group(y)
            if counts[gy]*U<=q*ref[gy]:continue
            av=counts[gx]*U-q*ref[gx];bv=counts[gy]*U-q*ref[gy]
            err=base-abs(av)-abs(bv)+abs(av+U)+abs(bv-U)
            if err>=base:continue
            if dc[demo(y)]==1 and demo(x)!=demo(y):continue
            if endpoint(y) and demo(y) in protected and ec[demo(y)]==1:continue
            totals['support_and_E_legal']+=1
            r=float(max_by[slot]);m=float(mean_by[slot])
            badr=r>R0+1e-12;badm=m>M0+1e-12
            totals['fails_both' if badr and badm else 'fails_radius_only' if badr else 'fails_mean_only' if badm else 'coverage_legal']+=1
            if not badr and not badm:legal.append((err,r,m,ids[x],ids[y]))
    if not legal:return None,dict(totals)
    e=min(v[0] for v in legal);legal=[v for v in legal if v[0]==e]
    r=min(v[1] for v in legal);legal=[v for v in legal if v[1]<=r+1e-12]
    m=min(v[2] for v in legal);legal=[v for v in legal if v[2]<=m+1e-12]
    return min(legal,key=lambda v:(v[3],v[4]))[3:],dict(totals)

def main():
    start=time.time();out=BASE/'INDEPENDENT_GEOMETRY.json';assert not out.exists()
    inputs=json.loads((BASE/'INPUT_HASHES.json').read_text())
    result_hashes={str(p):sha(p) for p in BASE.glob('task*.json')}
    assert all(sha(p)==h for p,h in inputs.items())
    spec=importlib.util.spec_from_file_location('parent_distance',SRC);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    features=json.loads((P/'FEATURES.json').read_text());results=[]
    for t in range(6,10):
        r=json.loads((BASE/f'task{t}.json').read_text());meta=r['metadata'];ids=sorted(meta);idx={n:i for i,n in enumerate(ids)}
        b=json.loads((SC/'bindings'/f'task{t}.json').read_text())
        with h5py.File(b['hdf5']) as f:pop=np.concatenate([f['data'][d]['actions'][:] for d in features['demos'][str(t)]])
        v=[];a=[]
        for name in ids:
            obj=torch.load(P/'features'/str(t)/name,map_location='cpu')
            v.append(obj['feature'][0,:,1:3,:].numpy());a.append(obj['actions'][0,-1,:].numpy())
        D=mod.Distance(v,a,pop,42).distances(np.arange(len(ids)))
        assert hashlib.sha256(D.tobytes()).hexdigest()==r['D_sha256']
        seq=[idx[s] for s in r['original_ids']];ref={int(g):n for g,n in r['reference_unique_counts'].items()}
        for w in r['swaps']:
            best,counts=check_search(seq,D,ids,meta,ref,r['before']['radius'],r['before']['mean'],set(r['before']['endpoint_demos']))
            assert best==(w['added'],w['removed']),(t,w['step'],best)
            assert counts['support_and_E_legal']==w['search']['support_and_E_legal']
            assert counts.get('coverage_legal',0)==w['search']['coverage_legal']
            seq[w['slot']]=idx[w['added']]
            nearest=D[:,seq].min(1)
            assert abs(float(nearest.max())-w['after']['radius'])<1e-12
            assert abs(float(nearest.mean())-w['after']['mean'])<1e-12
        best,counts=check_search(seq,D,ids,meta,ref,r['before']['radius'],r['before']['mean'],set(r['before']['endpoint_demos']))
        assert best is None and counts['support_and_E_legal']==r['terminal_search']['support_and_E_legal']
        results.append(dict(task=t,status='PASS',all_greedy_choices_reproduced=len(r['swaps']),terminal_counts=counts))
        print(json.dumps(results[-1]),flush=True)
    assert all(sha(p)==h for p,h in inputs.items())
    assert all(sha(p)==h for p,h in result_hashes.items())
    value=dict(status='PASS',seconds=time.time()-start,tasks=results,
               inputs_unchanged=True,metadata_unchanged=True,optimizer_calls=0,environment_calls=0,
               code_sha256=sha(__file__),result_sha256=result_hashes)
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)

if __name__=='__main__':main()
