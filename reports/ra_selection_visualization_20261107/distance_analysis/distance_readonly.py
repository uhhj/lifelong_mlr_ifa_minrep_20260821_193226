CASES=[{'suite': 'OBJECT', 'seed': 41, 'remote': '/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019', 'expected': {'FEATURES.json': 'b9750381b138b9425266f58152a434fc85a837addadb53aabc92d78ee176e750', 'ra_coreset/storage/task6.json': 'eee50d54f8fcad2f9810a55c5571e5607d1d8c2e0211ddb25b6f86d1c7e53df3', 'ra_coreset/storage/task7.json': '1dcc3fb23139a8ff2ce815f04ae9d4a21d9fe094358a829ac17b3abe51e6b29e', 'ra_coreset/storage/task8.json': '16ef36ac7b181f99b97df015984bf634c6d463520dc280d8b6af3f749a1cfc73', 'ra_coreset/storage/task9.json': '0aedb9f70505f21701c549383c76dd550a3cd14e34a91e62ae07f2a56382bfc1'}}, {'suite': 'OBJECT', 'seed': 42, 'remote': '/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019', 'expected': {'FEATURES.json': 'beeee2a789e8c28c07c00230a69f1206dc50dd5c87a71bff852916ac656c7c41', 'ra_coreset/storage/task6.json': '380535fa0c80162362e202be714c93e19d5733d694ce7b92ea6e8209b90f6ef1', 'ra_coreset/storage/task7.json': 'b4852187de769cf2d194c90bdb77b8e73007c0c7fd564af03f1097b48ba06cc8', 'ra_coreset/storage/task8.json': '425d8398a6d41e5db573d671801d1af034c13f99e4cf441e0823680ecb3e87e3', 'ra_coreset/storage/task9.json': '8bb5958343eea622339f203773661e90a00a6e947eb62ce23f713e46f478b1f5'}}, {'suite': 'OBJECT', 'seed': 43, 'remote': '/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed43_20261019', 'expected': {'FEATURES.json': 'fa491121ce284eba584d8ff4922f854d6fad38e509cffd63f67ed98766163ba3', 'ra_coreset/storage/task6.json': 'aa376eae2b9508ab004b254ced9a79c533a547362a3ccec110a7b5948dc5a006', 'ra_coreset/storage/task7.json': 'a947c6d4d4bb395675f276b6372499ab90653e02055f52e079412aeab634bf89', 'ra_coreset/storage/task8.json': '32fa8e938055bb4e5c7b8931a1ba27c5f16ddfe7f5e2e6d504f8c66f55980df3', 'ra_coreset/storage/task9.json': '4b7ea2381088214a9ee0e50dbe6fff69f25bfe804893307216f7315b9b8ca38d'}}, {'suite': 'GOAL', 'seed': 41, 'remote': '/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019', 'expected': {'FEATURES.json': '44c5824eddd3505ea596daace874b6752d3678dbe0b770c532ff1acd3c2dd4ea', 'ra_coreset/storage/task6.json': '869c3c273d271bd971e537c4d041fc51b8a319664d4359aa3dfc264d4b7f2a00', 'ra_coreset/storage/task7.json': '282ead88105bd4894baf73514867a33f5ecdfb4ae4fb28314dec43e0858cc25f', 'ra_coreset/storage/task8.json': 'bad2921ab01aa9761087d7861fdc4c5b7af9d4a1217696729986830c9cfb7cbb', 'ra_coreset/storage/task9.json': '444655af06f101efcfdfcf4dc71d904d5ccd1b077082c8657a7f67e9e4ebd94c'}}, {'suite': 'GOAL', 'seed': 42, 'remote': '/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018', 'expected': {'FEATURES.json': 'f494a8c7e7be81f4f8e562d83b7b9af450b6061b5e3bb0d1cb5ad5c0c1d78a6b', 'ra_coreset/storage/task6.json': '418ca10f13bdfebcf0bbcc64548068c6226a49fd5e2215004debaaacbc3818ff', 'ra_coreset/storage/task7.json': '75336a56e6726829b2570f8e03269c597918cc820d51bb20581200198196fbd3', 'ra_coreset/storage/task8.json': 'c7c4a52e070c125480fd76b3019c5ff6b6899604274f345b962672e9a54306e0', 'ra_coreset/storage/task9.json': '9be3d9dfb25e4ac46ac1f11b3ac12ead85a4eda84ba1abcf20bf1578c1567b8b'}}, {'suite': 'GOAL', 'seed': 43, 'remote': '/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed43_20261019', 'expected': {'FEATURES.json': '2b0b0074ae0105db8de075972d71126a94cad8a316d8def579ab013b30df02e8', 'ra_coreset/storage/task6.json': 'b1ff4912724eb3c689b6e03c91b8a1e33bcaa57a6d3f0cddd77cdfd65a1dae46', 'ra_coreset/storage/task7.json': '325524b6b8d3361503e40e8b105c7cae341baa8a6f9e9a1d5a3f2f004781108e', 'ra_coreset/storage/task8.json': '4b3fc40cbc5cfde513d1542db7827db645036fb0e0530cad5ab9652c9f2f8226', 'ra_coreset/storage/task9.json': '5b8a53fc81da09b98448dac63e510fb208278e4c029f4ddb6d9580300c5c383d'}}]
FROZEN_SOURCE='"""Frozen exploratory distance; no global RNG or evaluation information."""\nimport random\nimport time\nfrom pathlib import Path\nimport numpy as np\n\nSETTINGS = dict(version=\'visual_action_kcenter_v1\', vision_tokens=[1, 2],\n    time_positions=8, visual_weight=0.5, action_weight=0.5, calibration_pairs=4096,\n    reference_count=256, seed=42, block_size=256, numeric_dtype=\'float64\',\n    action_scale=\'MAD*1.4826; then IQR/1.349; then range; then 1\',\n    action_distance=\'RMS standardized last-step action difference\',\n    distance_scale=\'median strictly positive calibration distances, or 1 if all zero\',\n    center=\'minimum mean combined distance to seeded reference subset\',\n    tie_break=\'lexicographic filename; absolute float64 tie tolerance 1e-12\', clipping=False)\n\ndef robust_scale(actions):\n    a=np.asarray(actions,dtype=np.float64)\n    med=np.median(a,axis=0)\n    candidates=[np.median(np.abs(a-med),axis=0)*1.4826,\n                (np.percentile(a,75,axis=0)-np.percentile(a,25,axis=0))/1.349,\n                np.ptp(a,axis=0),np.ones(a.shape[1])]\n    out=candidates[0].copy()\n    for c in candidates[1:]:out=np.where(out>1e-12,out,c)\n    return out\n\ndef reference_pairs(n,seed=42,limit=4096):\n    total=n*(n-1)//2\n    flat=sorted(random.Random(seed).sample(range(total),min(limit,total)))\n    rows=[];i=0;offset=0\n    for q in flat:\n        while q>=offset+n-i-1:offset+=n-i-1;i+=1\n        rows.append((i,i+1+q-offset))\n    return np.asarray(rows,dtype=np.int64).reshape(-1,2)\n\nclass Distance:\n    def __init__(self,vision,actions,action_population,seed=42,block=256):\n        v=np.asarray(vision,dtype=np.float64);a=np.asarray(actions,dtype=np.float64)\n        assert v.ndim==4 and v.shape[1:3]==(8,2) and a.shape==(len(v),7)\n        assert len(v)>0 and np.isfinite(v).all() and np.isfinite(a).all()\n        norms=np.linalg.norm(v,axis=-1,keepdims=True)\n        assert (norms>1e-12).all(),\'Undefined cosine: zero visual token\'\n        self.v=(v/norms).reshape(len(v),-1);self.block=block\n        self.action_scale=robust_scale(action_population)\n        self.a=a/self.action_scale\n        self.pairs=reference_pairs(len(v),seed)\n        if len(self.pairs):\n            i,j=self.pairs.T\n            vh=np.maximum(0,1-np.sum(self.v[i]*self.v[j],axis=1)/16)\n            da=np.sqrt(np.mean((self.a[i]-self.a[j])**2,axis=1))\n        else:vh=da=np.zeros(0)\n        self.hs=float(np.median(vh[vh>1e-12])) if np.any(vh>1e-12) else 1.\n        self.as_=float(np.median(da[da>1e-12])) if np.any(da>1e-12) else 1.\n    def distances(self,ids):\n        ids=np.asarray(ids,dtype=np.int64);out=np.empty((len(self.v),len(ids)))\n        for start in range(0,len(self.v),self.block):\n            end=min(start+self.block,len(self.v))\n            vh=np.maximum(0,1-self.v[start:end]@self.v[ids].T/16)\n            da=np.sqrt(np.mean((self.a[start:end,None,:]-self.a[ids][None,:,:])**2,axis=-1))\n            out[start:end]=.5*vh/self.hs+.5*da/self.as_\n        out[ids,np.arange(len(ids))]=0.\n        return out\n\ndef select_arrays(ids,vision,actions,action_population,k,seed=42,block=256):\n    assert len(ids)==len(set(ids)) and 1<=k<=len(ids)\n    order=np.argsort(np.asarray(ids));ids=[ids[i] for i in order]\n    d=Distance(np.asarray(vision)[order],np.asarray(actions)[order],action_population,seed,block)\n    refs=sorted(random.Random(seed).sample(range(len(ids)),min(256,len(ids))))\n    centrality=d.distances(refs).mean(axis=1)\n    center=int(np.flatnonzero(centrality<=centrality.min()+1e-12)[0])\n    chosen=[center];nearest=d.distances([center])[:,0];radii=[float(nearest.max())]\n    while len(chosen)<k:\n        nearest[chosen]=-np.inf\n        nxt=int(np.flatnonzero(nearest>=nearest.max()-1e-12)[0]);chosen.append(nxt)\n        nearest=np.minimum(nearest,d.distances([nxt])[:,0])\n        rest=np.ones(len(ids),dtype=bool);rest[chosen]=False\n        radii.append(float(nearest[rest].max()) if rest.any() else 0.)\n    return [ids[i] for i in chosen],dict(settings=SETTINGS,initial_center=ids[center],\n        reference_ids=[ids[i] for i in refs],calibration_pair_ids=[[ids[i],ids[j]] for i,j in d.pairs],\n        action_scale=d.action_scale.tolist(),visual_scale=d.hs,action_distance_scale=d.as_,\n        covering_radius=radii)\n\ndef greedy_kcenter_select(paths,k,seed=42,action_population=None):\n    import torch\n    begin=time.monotonic();paths=sorted(paths,key=lambda p:Path(p).name)\n    assert action_population is not None,\'Raw current-task demonstration actions required\'\n    vv=[];aa=[]\n    for path in paths:\n        x=torch.load(path,map_location=\'cpu\');v=x[\'feature\'].detach().cpu().numpy();a=x[\'actions\'].detach().cpu().numpy()\n        assert v.shape==(1,8,4,512) and a.shape==(1,8,7)\n        vv.append(v[0,:,1:3,:]);aa.append(a[0,-1,:])\n    names,report=select_arrays([Path(p).name for p in paths],vv,aa,action_population,k,seed)\n    lookup={Path(p).name:p for p in paths};report[\'seconds\']=time.monotonic()-begin\n    return [lookup[n] for n in names],report\n'
FROZEN_FILE_SHA256='7c6640474c786fb73ee6b9aaedfa04a152280880489010d69e014ebe2430610e'
"""Executed with frozen Distance source and expected manifest hashes prefixed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import json,hashlib,time,random,inspect
from pathlib import Path
from collections import Counter
import numpy as np
import torch,h5py
torch.set_num_threads(4)
scope={};exec(compile(FROZEN_SOURCE,'archived_greedy_kcenter.py','exec'),scope)
Distance=scope['Distance']
hashes={};results=[];diagnostics={};begin=time.time()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def track(p):
    h=sha(p)
    if str(p) in hashes:assert hashes[str(p)]==h
    hashes[str(p)]=h;return h
def read(p):track(p);return json.loads(p.read_text())
def stats(a):return dict(mean=float(np.mean(a)),p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(np.max(a)),zero_fraction=float(np.mean(a==0)))
for case in CASES:
    root=Path(case['remote']);suite=case['suite'];seed=case['seed']
    for name,expected in case['expected'].items():assert track(root/name)==expected,(suite,seed,name)
    fm=read(root/'FEATURES.json');tasks=read(root/'TASKS.json')
    for task in range(6,10):
        start=time.time();key=f'{suite}_{seed}_{task}'
        m=read(root/f'ra_coreset/storage/task{task}.json');s=m['selection'];bfile=root/f'bindings/task{task}.json';binding=read(bfile)
        files=sorted((root/'features'/str(task)).glob('*.pt'),key=lambda p:p.name);names=[p.name for p in files];lookup={n:i for i,n in enumerate(names)}
        assert len(names)==s['population']==fm['counts'][str(task)]
        assert set(names)==set(binding['windows'])
        vv=[];aa=[]
        for p in files:
            assert track(p)==fm['hashes'][f'{task}/{p.name}']==binding['windows'][p.name]['sha256']
            x=torch.load(p,map_location='cpu')
            v=x['feature'].detach().numpy();a=x['actions'].detach().numpy()
            assert v.shape==(1,8,4,512) and a.shape==(1,8,7)
            assert np.array_equal(a[0,-1,:],np.asarray(binding['windows'][p.name]['last_action'],dtype=np.float32))
            vv.append(v[0,:,1:3,:].copy());aa.append(a[0,-1,:].copy())
        hdf=Path(tasks[str(task)]['paths'][0]);assert track(hdf)==binding['hdf5_sha256']
        with h5py.File(hdf,'r') as f:
            population=np.concatenate([f['data'][d]['actions'][:] for d in fm['demos'][str(task)]])
        original=s['original_selection'];settings=original['settings'];assert settings==scope['SETTINGS']
        dist=Distance(vv,aa,population,seed=settings['seed'])
        assert np.allclose(dist.action_scale,original['action_scale'],rtol=0,atol=1e-12)
        assert abs(dist.hs-original['visual_scale'])<1e-12 and abs(dist.as_-original['action_distance_scale'])<1e-12
        assert [[names[i],names[j]] for i,j in dist.pairs]==original['calibration_pair_ids']
        D=dist.distances(np.arange(len(names)));assert np.isfinite(D).all() and D.min()>=0
        assert np.max(np.abs(D-D.T))<1e-12 and np.all(np.diag(D)==0)
        draws=random.Random(seed).sample(names,s['quota']);assert draws==s['random_draw_ids']
        methods={'Random':draws,'K-Center':s['original_ids'],'RA-Coreset':s['selected_ids']}
        assert len(s['anchor_ids'])==s['quota']//2 and set(s['anchor_ids'])<=set(s['selected_ids'])
        nearest={};arms={};sets={};democounts={}
        for method,chosen in methods.items():
            assert len(chosen)==len(set(chosen))==s['quota'];idx=[lookup[n] for n in chosen];sets[method]=set(chosen)
            nearest[method]=D[:,idx].min(axis=1);assert np.all(nearest[method][idx]==0)
            arms[method]=stats(nearest[method]);democounts[method]=dict(Counter(binding['windows'][n]['demo_id'] for n in chosen))
        for method,before_after in [('K-Center','before'),('RA-Coreset','after')]:
            assert abs(arms[method]['mean']-s['mean_'+before_after])<1e-10
            assert abs(arms[method]['maximum']-s['max_'+before_after])<1e-10
        assert democounts['K-Center']==s['demo_original'] and democounts['RA-Coreset']==s['demo_final']
        assert abs(arms['K-Center']['maximum']-original['covering_radius'][-1])<1e-10
        rows=[]
        for i,n in enumerate(names):
            b=binding['windows'][n]
            rows.append(dict(window_id=n,demo=b['demo_id'],target=b['target'],length=b['length'],random_selected=n in sets['Random'],kcenter_selected=n in sets['K-Center'],ra_selected=n in sets['RA-Coreset'],anchor=n in s['anchor_ids'],random_distance=float(nearest['Random'][i]),kcenter_distance=float(nearest['K-Center'][i]),ra_distance=float(nearest['RA-Coreset'][i])))
        diagnostics['nearest_'+key]=rows
        projection=None
        if seed==41 and task==6:
            # Same precomputed original-distance matrix, one common projection for every subset.
            # Only rendering coordinates leave the server, never original visual/action tensors.
            from sklearn.manifold import TSNE
            import sklearn
            kwargs=dict(n_components=2,perplexity=30.0,early_exaggeration=12.0,learning_rate=200.0,metric='precomputed',init='random',random_state=0,method='barnes_hut',angle=.5,verbose=0)
            params=inspect.signature(TSNE).parameters
            kwargs['max_iter' if 'max_iter' in params else 'n_iter']=1000
            model=TSNE(**kwargs);xy=model.fit_transform(D.copy())
            assert xy.shape==(len(names),2) and np.isfinite(xy).all()
            diagnostics['projection_'+key]=[dict(window_id=n,x=float(xy[i,0]),y=float(xy[i,1])) for i,n in enumerate(names)]
            projection=dict(method='t-SNE',sklearn=sklearn.__version__,parameters=kwargs,kl_divergence=float(model.kl_divergence_),interpretation='Illustration only; not a coverage metric; common projection for all subsets.')
        results.append(dict(suite=suite,seed=seed,task=task,N=len(names),K=s['quota'],arms=arms,demo_counts=democounts,full_demo_counts=dict(Counter(v['demo_id'] for v in binding['windows'].values())),projection=projection,seconds=time.time()-start))
        del D,vv,aa,dist
assert all(sha(Path(p))==h for p,h in hashes.items())
diagnostics['input_hashes']=[dict(path=p,sha256=h) for p,h in hashes.items()]
print(json.dumps(dict(status='PASS_ALL_24_ORIGINAL_DISTANCE_RECOMPUTATIONS',observed=time.time(),results=results,diagnostics=diagnostics,frozen_distance_source_sha256=FROZEN_FILE_SHA256,inputs_before_after_unchanged=True,checked_files=len(hashes),seconds=time.time()-begin,optimizer_calls=0,environment_steps=0,model_forward_calls=0,selection_calls=0,device='CPU only')))
