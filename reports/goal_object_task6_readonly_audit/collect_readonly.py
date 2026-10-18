"""CPU-only existing-payload statistics. No model, training, export or environment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib,io,json,time,collections,sys
from pathlib import Path
import numpy as np
import torch
import h5py

P=Path(__file__).resolve().parent
HASH={}; expected={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):
    p=Path(p);b=p.read_bytes();HASH[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def desc(a):
    a=np.asarray(a,dtype=np.float64);tr=np.sqrt((a[:,:3]**2).mean(1))
    delta=np.sqrt((np.diff(a,axis=0)**2).mean(0)) if len(a)>1 else np.zeros(7)
    return np.r_[a[-1],a.mean(0),a[-1]-a[0],delta,np.mean(tr*(a[:,-1]>0)),np.mean(tr*(a[:,-1]<0))]
NAMES=[g+'_'+a for g in ('last','mean','net_change','delta_rms') for a in ('dx','dy','dz','rx','ry','rz','grip')]+['close_x_translation_rms','open_x_translation_rms']
def summaries(v,ids):
    nearest=v[:,ids].min(1);unselected=np.ones(len(v),bool);unselected[ids]=False
    return dict(max=float(nearest.max()),mean=float(nearest.mean()),p95=float(np.quantile(nearest,.95)),unselected_mean=float(nearest[unselected].mean()))
def main():
    assert not (P/'DIAGNOSTICS.json').exists(),'No overwrite'
    torch.set_num_threads(1);started=time.time();out={'status':'COLLECTING','cells':[],'metadata':{},'descriptor_names':NAMES}
    for suite,seeds in [('GOAL',(42,43,44)),('OBJECT',(40,41,42,43,44))]:
      for seed in seeds:
        if suite=='GOAL':
            root=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed%d_20261017'%seed);binding=None
        elif seed<42:
            base=Path('/data/mlr_ifa/experiments/Seeds40_41_three_arm_20261016/seed40' if seed==40 else '/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41')
            root=base/'rc';binding=read(base/'ec/WINDOW_BINDING.json')
        else:
            root=Path('/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed%d_20261013'%seed)
            binding=read(Path('/data/mlr_ifa/experiments/Event_Coreset_OBJECT_seed%d_20261015'%seed)/'WINDOW_BINDING.json')
        contract=read(root/'CONTRACT.json');fm=read(root/'FEATURES.json');tasks=read(root/'TASKS.json');read(root/'MODAL_CONTRACT.json')
        stores={a:read(root/a/'storage/task6.json') for a in ('random','coreset')}
        ifa={a:{str(t):read(root/a/('ifa_task%d.json'%t)) for t in range(6,10)} for a in ('random','coreset')}
        hdf=Path(tasks['6']['paths'][0]);h=sha(hdf);HASH[str(hdf)]=h;assert contract['inputs'][str(hdf)]==h
        with h5py.File(hdf,'r') as f:
            acts={d:f['data'][d]['actions'][:].astype(np.float32) for d in fm['demos']['6']}
        candidates=collections.defaultdict(list)
        for d,a in acts.items():
            for start in range(len(a)):
                inds=np.minimum(np.arange(start,start+8),len(a)-1);key=a[inds].tobytes()
                candidates[key].append({'demo':d,'start':start,'target':int(inds[-1]),'valid':min(8,len(a)-start),'length':len(a)})
        bind={r['sample_id']:r for r in binding['windows'] if r['task']==6} if binding else None
        names=sorted(p.name for p in (root/'features/6').glob('*.pt'));assert len(names)==fm['counts']['6']
        vv=[];state=[];last=[];dd=[];windows=[]
        for name in names:
            p=root/'features/6'/name;b=p.read_bytes();h=hashlib.sha256(b).hexdigest();HASH[str(p)]=h
            assert h==fm['hashes']['6/'+name]
            x=torch.load(io.BytesIO(b),map_location='cpu');a=x['actions'].numpy()[0];feat=x['feature'].numpy()[0]
            assert a.shape==(8,7) and feat.shape==(8,4,512)
            matches=candidates[a.astype(np.float32).tobytes()];assert matches,(suite,seed,name)
            if bind:
                w=bind[name];assert w['original_file_sha256']==h and w['exact_actions']
                matches=[m for m in matches if m['demo']==w['demo_id'] and m['start']==w['start']];assert len(matches)==1
            valid=matches[0]['valid']
            descriptors=[desc(a[:m['valid']]) for m in matches]
            descriptor_known=all(np.array_equal(descriptors[0],v) for v in descriptors)
            windows.append(dict(id=name,sha256=h,candidates=matches,unique=len(matches)==1))
            dd.append(descriptors[0] if descriptor_known else np.full(30,np.nan));vv.append(feat[:,1:3,:]);state.append(feat[:,3,:]);last.append(a[-1])
        v=np.asarray(vv,dtype=np.float64);v=(v/np.linalg.norm(v,axis=-1,keepdims=True)).reshape(len(names),-1)
        st=np.asarray(state,dtype=np.float64);st=(st/np.linalg.norm(st,axis=-1,keepdims=True)).reshape(len(names),-1)
        aa=np.asarray(last,dtype=np.float64);sel=stores['coreset']['selection']
        va=np.maximum(0,1-v@v.T/16);sa=np.maximum(0,1-st@st.T/8)
        scaled=aa/np.array(sel['action_scale']);ad=np.sqrt(((scaled[:,None,:]-scaled[None,:,:])**2).mean(-1))
        combined=.5*va/sel['visual_scale']+.5*ad/sel['action_distance_scale']
        for dist in (va,sa,ad,combined):np.fill_diagonal(dist,0)
        X=np.asarray(dd);sd=np.nanstd(X,axis=0);active=sd>1e-12;z=(X-np.nanmean(X,axis=0))/np.where(active,sd,1)
        idx={n:i for i,n in enumerate(names)};ids={a:[idx[Path(r['source']).name] for r in stores[a]['selected']] for a in stores}
        for a,items in stores.items():
            assert len(set(ids[a]))==items['K']==int(np.ceil(len(names)/2))
            for w in items['selected']:assert w['sha256']==HASH[str(root/'features/6'/Path(w['source']).name)]
        # Native selector's terminal radius is max distance of unselected population.
        assert abs(summaries(combined,ids['coreset'])['max']-sel['covering_radius'][-1])<1e-8
        switches={d:[i for i in range(1,len(a)) if a[i,-1]*a[i-1,-1]<0] for d,a in acts.items()}
        def profile(indices):
            ww=[windows[i]['candidates'][0] for i in indices if windows[i]['unique']]
            demos=collections.Counter(w['demo'] for w in ww)
            quart=collections.Counter(min(3,int(w['target']/max(1,w['length']-1)*4)) for w in ww)
            points={d:set(w['target'] for w in ww if w['demo']==d) for d in acts}
            event=[];gaps={}
            for d,pp in points.items():
                coords=sorted(pp);gaps[d]=max(np.diff([-1]+coords+[len(acts[d])]).tolist())-1
                for t in switches[d]:event.append({'demo':d,'step':t,'pre':any(t-j in pp for j in range(1,5)),'post':any(t+j in pp for j in range(1,5))})
            return dict(n=len(indices),unique_position_n=len(ww),demo_counts=dict(demos),quartile_counts=dict(quart),padding_n=sum(w['valid']<8 for w in ww),
                near_switch_n=sum(any(abs(w['target']-t)<=4 for t in switches[w['demo']]) for w in ww),
                events=event,events_both_sides=sum(v['pre'] and v['post'] for v in event),events_total=len(event),max_missing_target_run=gaps,
                positive_gripper_fraction=float((aa[indices,-1]>0).mean()),negative_gripper_fraction=float((aa[indices,-1]<0).mean()),
                translation_rms_mean=float(np.sqrt((aa[indices,:3]**2).mean(1)).mean()),rotation_rms_mean=float(np.sqrt((aa[indices,3:6]**2).mean(1)).mean()),
                descriptor_known_n=int(np.isfinite(z[indices]).all(1).sum()),standardized_descriptor_mean=np.nanmean(z[indices],axis=0).tolist(),descriptor_mean_bias_rms=float(np.sqrt((np.nanmean(z[indices][:,active],axis=0)**2).mean())))
        methods={}
        for a,ii in ids.items():
            methods[a]=dict(profile=profile(ii),selected_ids=[names[i] for i in ii],coverage={n:summaries(dist,ii) for n,dist in [('combined',combined),('visual',va),('action',ad),('state',sa)]})
        full=profile(list(range(len(names))))
        for a in methods:
            prof=methods[a]['profile'];nn=prof['unique_position_n'];nf=full['unique_position_n']
            prof['demo_total_variation']=sum(abs(prof['demo_counts'].get(d,0)/nn-full['demo_counts'].get(d,0)/nf) for d in acts)/2
            prof['quartile_total_variation']=sum(abs(prof['quartile_counts'].get(i,0)/nn-full['quartile_counts'].get(i,0)/nf) for i in range(4))/2
        for i,w in enumerate(windows):
            w['selected']={a:i in set(ii) for a,ii in ids.items()}
        cell=dict(suite=suite,seed=seed,root=str(root),task_name=tasks['6']['name'],N=len(names),K=stores['random']['K'],
            unique_position_count=sum(w['unique'] for w in windows),ambiguous=[w for w in windows if not w['unique']],
            window_bindings=windows,population=full,methods=methods,member_overlap=len(set(ids['random'])&set(ids['coreset'])),
            ifa=ifa,gamma=contract['gamma'],calibration={k:sel[k] for k in ('action_scale','visual_scale','action_distance_scale')})
        out['cells'].append(cell)
        print(json.dumps({'suite':suite,'seed':seed,'N':len(names),'unique':cell['unique_position_count'],'seconds':time.time()-started}),flush=True)
    assert all(sha(p)==h for p,h in HASH.items()),'Input modified during audit'
    out.update(status='PASS_READONLY',inputs_sha256=HASH,hashes_before_after_equal=True,seconds=time.time()-started,
        calls={'optimizer':0,'model_forward':0,'environment':0,'new_feature_export':0},torch=torch.__version__,numpy=np.__version__)
    (P/'DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'cells':len(out['cells']),'inputs':len(HASH),'seconds':out['seconds']}),flush=True)
if __name__=='__main__':main()
