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
