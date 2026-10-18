"""CPU-only existing-window selection; writes only new SC-v2 metadata.
No checkpoint loading, model forward, optimizer, storage copy, or environment.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json,hashlib,time,importlib.util,sys
from pathlib import Path
import numpy as np
import torch,h5py
from selection import repair
torch.set_num_threads(2)
SC=Path('/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018')
P=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017')
SOURCE=Path('/root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/source')
OUT=Path('/root/mlr_ifa/analyses/SCv2_GOAL_seed42_offline_20261018_verified')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    begin=time.time();files={}
    selector=SOURCE/'libero/lifelong/selectors/greedy_kcenter.py'
    spec=importlib.util.spec_from_file_location('original_selector',selector)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for p in [selector,SC/'CONTRACT.json',SC/'MODAL_CONTRACT.json',P/'CONTRACT.json',
              P/'MODAL_CONTRACT.json',P/'FEATURES.json']:
        files[str(p)]=sha(p)
    features=json.loads((P/'FEATURES.json').read_text())
    summaries=[]
    for t in range(6,10):
        started=time.time();print(f'Task {t}: verifying real inputs',flush=True)
        binding_file=SC/'bindings'/f'task{t}.json'
        previous_file=SC/'selection_preflight'/f'task{t}.json'
        parent_file=P/'coreset/storage'/f'task{t}.json'
        for p in [binding_file,previous_file,parent_file]:files[str(p)]=sha(p)
        binding=json.loads(binding_file.read_text());rows=binding['windows']
        previous=json.loads(previous_file.read_text());parent=json.loads(parent_file.read_text())
        hdf=Path(binding['hdf5']);files[str(hdf)]=sha(hdf)
        assert files[str(hdf)]==binding['hdf5_sha256']
        paths=sorted((P/'features'/str(t)).glob('*.pt'),key=lambda p:p.name)
        ids=[p.name for p in paths];assert set(ids)==set(rows) and len(ids)==features['counts'][str(t)]
        vision=[];actions=[]
        with h5py.File(hdf,'r') as f:
            demos=features['demos'][str(t)]
            raw={d:f['data'][d]['actions'][:] for d in demos}
        for p in paths:
            files[str(p)]=sha(p);r=rows[p.name];assert files[str(p)]==r['sha256']
            obj=torch.load(p,map_location='cpu');v=obj['feature'];a=obj['actions']
            assert tuple(v.shape)==(1,8,4,512) and tuple(a.shape)==(1,8,7)
            aa=a[0].numpy();rr=raw[r['demo_id']]
            assert len(rr)==r['length']
            assert np.array_equal(aa[-1],np.asarray(r['last_action'],dtype=np.float32))
            # Existing binding/feature export defines exact float32 action identity.
            # Keep raw population precision below unchanged for original distance scaling.
            rr32=rr.astype(np.float32)
            if p==paths[0]:
                print(json.dumps(dict(task=t,cache_dtype=str(aa.dtype),hdf_dtype=str(rr.dtype),
                      raw_max_delta=float(np.max(np.abs(aa[-1]-rr[r['target']]))),
                      float32_exact=bool(np.array_equal(aa[-1],rr32[r['target']])))),flush=True)
            assert np.array_equal(aa[-1],rr32[r['target']])
            for start in r['possible_starts']:
                expected=rr32[np.minimum(np.arange(start,start+8),len(rr)-1)]
                assert np.array_equal(aa,expected)
                assert min(start+7,len(rr)-1)==r['target']
            sign=0 if aa[-1,-1]<0 else (1 if aa[-1,-1]==0 else 2)
            assert sign==r['group']%3
            vision.append(v[0,:,1:3,:].numpy());actions.append(aa[-1])
        population=np.concatenate([raw[d] for d in demos])
        q=(len(ids)+1)//2
        print(f'Task {t}: recomputing unchanged original K-Center on CPU',flush=True)
        original,origreport=module.select_arrays(ids,vision,actions,population,q,42)
        assert original==previous['original_ids']
        assert original==[Path(r['source']).name for r in parent['selected']]
        d=module.Distance(vision,actions,population,42);D=d.distances(np.arange(len(ids)))
        assert d.hs==origreport['visual_scale'] and d.as_==origreport['action_distance_scale']
        Dhash=hashlib.sha256(D.tobytes()).hexdigest()
        print(f'Task {t}: exhaustive SC-v2 repair',flush=True)
        def progress(n,row):
            if n%10==0 or n==1:
                print(json.dumps(dict(task=t,swaps=n,E=row['E'],radius=row['radius'],mean=row['mean'],unique=row['unique_times'])),flush=True)
        result=repair(ids,rows,original,D,progress=progress)
        # Every selected swap already verified using direct D[:,chosen].min.
        result.update(task=t,N=len(ids),q=q,original_order_reproduced=True,
                      full_action_window_identity_verified=True,D_sha256=Dhash,
                      original_selector_sha256=sha(selector),original_geometry_settings=module.SETTINGS,
                      scales=dict(visual=d.hs,action=d.as_,action_dimensions=d.action_scale.tolist()),
                      total_seconds=time.time()-started,
                      metadata={n:{k:rows[n][k] for k in ['demo_id','target','length','group','sha256','possible_starts']} for n in ids})
        write(OUT/f'task{t}.json',result)
        summaries.append({k:result[k] for k in ['task','N','q','before','after','retained_original','total_seconds','terminal_search']})
        print(json.dumps(dict(task=t,status='PASS',swaps=len(result['swaps']),seconds=time.time()-started)),flush=True)
        del D,d,vision,actions,raw,population
    print('Verifying all input hashes after offline work',flush=True)
    after={p:sha(p) for p in files};assert files==after
    write(OUT/'INPUT_HASHES.json',files)
    write(OUT/'ACCEPTANCE.json',dict(status='OFFLINE_PASS',tasks=summaries,input_files=len(files),
          inputs_unchanged=True,optimizer_calls=0,model_forwards=0,checkpoint_loads=0,environment_calls=0,
          cpu_only=True,training_authorized_this_turn=False,evaluation_protocol='deferred by user',
          seconds=time.time()-begin,time=time.time(),
          code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print(json.dumps(dict(status='OFFLINE_PASS',output=str(OUT),seconds=time.time()-begin)),flush=True)

if __name__=='__main__':main()
