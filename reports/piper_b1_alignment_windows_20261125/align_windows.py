"""Offline B1 training restricted to pre-grasp, measured horizontal alignment windows."""
from bisect import bisect_right
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np


def setup(work):
    diag=work/'offline_diagnostics'
    sys.path[:0]=[str(diag/'b1_feedback_xy_20261125'),str(diag/'b1_spatial_ablation_20261121_2034')]
    import policy_xy as xy
    rel,_=xy.setup(work)
    return xy,rel,diag


def disjoint(candidates):
    """Maximize count, then net approach, with deterministic earlier-frame tie breaks."""
    ends=[r['t']+8 for r in candidates]
    scores=[(0,0.)]
    parent=[]
    for i,r in enumerate(candidates):
        earlier=bisect_right(ends,r['t'],hi=i)
        count,gain=scores[earlier]
        take=(count+1,gain+min(r['advance_mm'],r['contact_advance_mm']))
        if take>scores[-1]:
            scores.append(take)
            parent.append(earlier)
        else:
            scores.append(scores[-1])
            parent.append(None)
    use=set()
    at=len(candidates)
    while at:
        earlier=parent[at-1]
        if earlier is None:
            at-=1
        else:
            use.add(candidates[at-1]['t'])
            at=earlier
    return use


def select(work,out):
    xy,rel,diag=setup(work)
    from annotate import images,vectors
    from probe import load_model
    from train_contrast import MANIFEST,image_rgb
    import cv2
    import torch
    import torch.nn.functional as F
    prior,coord=rel.paths(work)
    manifest=diag/'b1_ode_lerobot_20261123/selection.json'
    assert rel.digest(manifest)==MANIFEST
    meta=json.loads(manifest.read_bytes())
    train=set(meta['selected_ids'])
    assert len(train)==50
    ranked={r['id']:r for r in meta['ranking']}
    labels=json.loads((prior/'points.json').read_bytes())
    manual=json.loads((prior/'annotations.json').read_bytes())
    geom=json.loads((prior/'final.json').read_bytes())['geometry']
    table=np.array(geom['table_map'])
    tip=np.array(geom['tip_mm'])
    net,sha=load_model(work,'baseline')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    encoder=net.obs_encoder
    backbone=encoder.key_model_map['overhead']
    all_eps,all_offsets,all_features,all_targets=[],[],[],[]
    summary={}
    for original in labels:
        r=dict(original)
        if r['episode'] not in train:continue
        r.update(manual.get(str(r['index']),{}))
        ep=r['episode']
        source=ranked[ep]
        folder=work/'batches/B1_20261123'/source['source']
        file=folder/'alignment.jsonl'
        assert rel.digest(file)==source['alignment_sha256']
        first=source['first']
        rows=[json.loads(line) for line in file.read_bytes().splitlines()][first:]
        q=vectors(rows,'feedback')
        stamp=np.array([s['feedback']['sample_monotonic_ns'] for s in rows],dtype=np.int64)
        xyz,_=xy.pose(q[:,:6],tip)
        assert r['contact'] is not None and r['uv'] is not None
        stop=min(k for k in (r['close'],r['contact']) if k is not None)
        block=np.r_[r['uv'],1.]@table
        contact=xyz[r['contact'],:2]
        distance=np.linalg.norm(xyz[:,:2]-block,axis=1)
        ref=np.linalg.norm(xyz[:,:2]-contact,axis=1)
        eligible=[]
        scanned=0
        for t in range(7,stop-8):
            end=t+8
            scanned+=1
            if not np.all(np.diff(stamp[t:end+1])>0):continue
            gain=distance[t]-distance[end]
            cross=ref[t]-ref[end]
            path=xyz[t:end+1]
            norm=np.linalg.norm(path[-1,:2]-path[0,:2])
            arc=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()
            height=np.ptp(path[:,2])
            if distance[t]>60 or min(gain,cross)<5 or gain<0.6*arc or norm<height:continue
            eligible.append(dict(t=t,source_start=first+t,source_end=first+end,
                start_sec=float((stamp[t]-stamp[0])/1e9),duration_ms=float((stamp[end]-stamp[t])/1e6),
                initial_mm=distance[t],final_mm=distance[end],advance_mm=gain,contact_advance_mm=cross,
                path_mm=arc,planar_chord_mm=norm,height_span_mm=height))
        use=disjoint(eligible)
        summary[ep]=dict(first_source_row=first,stop_exclusive=stop,scanned=scanned,
            contact_vs_image_mm=float(np.linalg.norm(block-contact)),qualifying=eligible)
        selected=sorted(use)
        for row in eligible:row['selected']=row['t'] in use
        if not selected:continue
        paths=images(folder,'overhead')[first:]
        with torch.inference_mode():
            for pos in range(0,len(selected),12):
                ticks=selected[pos:pos+12]
                pixels=np.stack([image_rgb(cv2.imread(str(paths[t])),128) for t in ticks])
                x=encoder.key_transform_map['overhead'](torch.from_numpy(pixels).float()/255)
                stage=backbone.layer2(backbone.layer1(backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))))
                feats=F.adaptive_avg_pool2d(stage,(8,8)).flatten(1).numpy()
                for t,feat in zip(ticks,feats):
                    all_features.append(feat)
                    all_targets.append(xyz[t+1:t+9,:2]-xyz[t,:2])
                    all_eps.append(ep)
                    all_offsets.append(stop-t)
    assert set(summary)==train
    np.savez(out/'private_selected.npz',episodes=all_eps,offsets=all_offsets,features=all_features,targets=all_targets)
    rel.dump(out/'selection.json',dict(episodes=summary,n_scanned=sum(r['scanned'] for r in summary.values()),
        n_qualifying=sum(len(r['qualifying']) for r in summary.values()),n_selected=len(all_eps),
        represented=len(set(all_eps)),protocol_sha256=rel.digest(out/'protocol.md'),checkpoint_sha256=sha,
        manifest_sha256=MANIFEST,selected_archive_sha256=rel.digest(out/'private_selected.npz')))
    print('scanned / qualifying / disjoint / demonstrations',sum(r['scanned'] for r in summary.values()),
          sum(len(r['qualifying']) for r in summary.values()),len(all_eps),len(set(all_eps)),flush=True)


def fit(work,out):
    xy,rel,diag=setup(work)
    _,coord=rel.paths(work)
    selected=json.loads((out/'selection.json').read_bytes())
    assert rel.digest(out/'private_selected.npz')==selected['selected_archive_sha256']
    with np.load(out/'private_selected.npz') as s:
        eps=s['episodes']
        offset=s['offsets']
        feature=s['features'].astype(float)
        y=s['targets'].reshape(-1,16)
    present=set(eps.tolist())
    all_train=set(selected['episodes'])
    readers=rel.coordinate_heads(coord,feature)
    cache={}
    def locations(allowed):
        key=frozenset(allowed)
        if key not in cache:
            cache[key]=readers[0].fit(key)[0]-readers[1].fit(key)[0]
        return cache[key]
    def crossfit(training_demos,allowed_labels):
        points=np.full((len(eps),2),np.nan)
        for val in rel.folds(training_demos):
            ids=np.isin(eps,sorted(val))
            points[ids]=locations(allowed_labels-val)[ids]
        return points
    losses={k:np.zeros(len(rel.PENALTIES)) for k in ('xy','permuted')}
    constant=0.
    for val in rel.folds(present):
        train=present-val
        allowed=all_train-val
        learn=np.flatnonzero(np.isin(eps,sorted(train)))
        valid=np.flatnonzero(np.isin(eps,sorted(val)))
        xx=crossfit(train,allowed)[learn]
        xv=locations(allowed)[valid]
        w=rel.weights(eps[valid])
        wt=rel.weights(eps[learn])
        def error(pred):
            return w@np.linalg.norm((pred-y[valid]).reshape(-1,8,2),axis=2).mean(1)
        constant+=error(wt@y[learn]/sum(wt))
        for mode,x in [('xy',xx),('permuted',rel.shuffle_by_demo(xx,eps[learn],offset[learn]))]:
            for i,p in enumerate(rel.PENALTIES):
                head=rel.fit_head(x,y[learn],eps[learn],p)
                losses[mode][i]+=error(rel.residual(head,xv))
    xx=crossfit(present,all_train)
    params={}
    fitted={}
    for mode,x in [('xy',xx),('permuted',rel.shuffle_by_demo(xx,eps,offset))]:
        reg=rel.PENALTIES[int(np.argmin(losses[mode]))]
        h=rel.fit_head(x,y,eps,reg)
        params.update({mode+'_'+k:v for k,v in h.items()})
        fitted[mode]=reg
    for reader,k in zip(readers,['wood','jaw']):
        pred,w,b,_=reader.fit(all_train)
        np.testing.assert_allclose(feature@w+b,pred,atol=1e-5)
        params[k+'_w'],params[k+'_b']=w,b
    # The frozen readout used at inference must be identical to the previous XY control.
    fixed_path=diag/'b1_feedback_xy_20261125/private_xy_policy.npz'
    with np.load(fixed_path) as prev:
        for k in ['wood_w','wood_b','jaw_w','jaw_b']:
            np.testing.assert_allclose(params[k],prev[k],rtol=1e-9,atol=1e-7)
    np.savez(out/'private_xy_policy.npz',**params)
    rel.dump(out/'fit.json',dict(candidate_penalties=rel.PENALTIES,selected_penalty=fitted,
        cv_mean_path_error_mm={k:(v/len(present)).tolist() for k,v in losses.items()},
        constant_cv_mean_path_error_mm=float(constant/len(present)),policy_sha256=rel.digest(out/'private_xy_policy.npz'),
        n_action_training=len(eps),n_demos=len(present),readout_training=[len(r.groups) for r in readers],
        matched_reference_readouts=True))
    rel.dump(out/'audit_inputs.json',dict(checkpoint_sha256=selected['checkpoint_sha256'],
        protocol_sha256=selected['protocol_sha256'],selection_sha256=rel.digest(out/'selection.json'),
        original_windows_sha256=json.loads((diag/'b1_relative_policy_20261125/data_audit.json').read_bytes())['private_archive_sha256']))
    print('training finished, penalties',fitted,flush=True)


def evaluation_context(work,out):
    xy,rel,diag=setup(work)
    _,_,rec,geo,goals,fk,d,active,start,target,_=xy.load_windows(work)
    audit=json.loads((out/'audit_inputs.json').read_bytes())
    paired=set(e for pair in geo['pairs'] for e in pair)
    checks=[]
    for i,ep in enumerate(d['episodes']):
        if ep in active:continue
        r=rec[ep]
        t=d['indices'][i]
        tip=np.array(geo['tip_mm'])
        p,_=xy.pose(r['q'][t:t+9,:6],tip)
        center=goals[ep]
        contact=fk(r['q'][r['contact']])
        delta=lambda g:float(np.linalg.norm(p[0,:2]-g)-np.linalg.norm(p[-1,:2]-g))
        gain,ref=delta(center),delta(contact)
        arc=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).sum()
        xy_norm=np.linalg.norm(p[-1,:2]-p[0,:2])
        stop=min(k for k in (r['close'],r['contact']) if k is not None)
        ok=(t+8<stop and np.linalg.norm(p[0,:2]-center)<=60 and min(gain,ref)>=5
            and gain>=.6*arc and xy_norm>=np.ptp(p[:,2]))
        checks.append(dict(episode=ep,paired=ep in paired,qualifies=bool(ok),advance_mm=gain))
    audit.update(pairs=geo['pairs'],heldout_episodes=geo['test_episodes'],fixed_anchor_context=checks)
    rel.dump(out/'audit_inputs.json',audit)
    print('all original evaluation anchors retained',len(checks),'match training criteria',sum(r['qualifies'] for r in checks),flush=True)


def phase_check(work,out):
    xy,rel,diag=setup(work)
    _,_,rec,geo,goals,fk,_,_,_,_,_=xy.load_windows(work)
    import cv2
    import torch
    import torch.nn.functional as F
    from probe import load_model
    from annotate import images
    from train_contrast import image_rgb
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    fit_record=json.loads((out/'fit.json').read_bytes())
    assert rel.digest(out/'private_xy_policy.npz')==fit_record['policy_sha256']
    with np.load(out/'private_xy_policy.npz') as archive:
        params=dict(archive)
    net,sha=load_model(work,'baseline')
    cached=[]
    def observe(module,inputs,output):
        cached.append(F.adaptive_avg_pool2d(output[-1:],(8,8)).flatten().numpy().astype(float))
    hook=net.obs_encoder.key_model_map['overhead'].layer2.register_forward_hook(observe)
    tip=np.asarray(geo['tip_mm'])
    @torch.inference_mode()
    def endpoints(obs,anchor):
        cached.clear()
        base=net.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].numpy()
        wood,jaw=xy.decoded(params,cached[0])
        orig=fk(obs['agent_pos'][0,-1].numpy())
        q,_=xy.joint_commands(base,orig+xy.displacement(params,'xy',wood,jaw),tip)
        return {'base':fk(base[-1]),'xy':fk(q[-1])}
    ticks={}
    for ep in sorted({e for pair in geo['pairs'] for e in pair}):
        r=rec[ep]
        p,_=xy.pose(r['q'][:,:6],tip)
        clock=np.array([json.loads(l)['feedback']['sample_monotonic_ns'] for l in
            (r['folder']/'alignment.jsonl').read_bytes().splitlines()[r['first']:]])
        stop=min(v for v in (r['close'],r['contact']) if v is not None)
        ref=p[r['contact'],:2]
        dist=np.linalg.norm(p[:,:2]-goals[ep],axis=1)
        contact=np.linalg.norm(p[:,:2]-ref,axis=1)
        valid=[]
        for t in range(7,stop-8):
            seg=p[t:t+9]
            gain=dist[t]-dist[t+8]
            arc=np.linalg.norm(np.diff(seg[:,:2],axis=0),axis=1).sum()
            if (dist[t]<=60 and min(gain,contact[t]-contact[t+8])>=5 and gain>=.6*arc and
                np.linalg.norm(seg[-1,:2]-seg[0,:2])>=np.ptp(seg[:,2]) and np.all(np.diff(clock[t:t+9])>0)):
                valid.append(t)
        ticks[ep]=min(valid,key=lambda t:(abs(t-(r['close']-18)),t)) if valid else None
    pairs=[p for p in geo['pairs'] if all(ticks[e] is not None for e in p)]
    hist={}
    original={}
    natural=[]
    directions=[]
    for ep in [e for pair in pairs for e in pair]:
        r=rec[ep]
        t=ticks[ep]
        obs={'agent_pos':torch.from_numpy(r['q'][None,t-7:t+1])}
        for view in ('overhead','wrist'):
            fs=images(r['folder'],view)[r['first']+t-7:r['first']+t+1]
            pixels=np.stack([image_rgb(cv2.imread(str(f)),128) for f in fs])
            obs[view]=torch.from_numpy(pixels[None]).float()/255
        anchor=torch.from_numpy(r['cmd'][None,t-1])
        hist[ep]=(obs,anchor)
        original[ep]=endpoints(obs,anchor)
        for mode,end in original[ep].items():
            natural.append(dict(episode=ep,mode=mode,error_mm=np.linalg.norm(end-fk(r['q'][t+8]))))
    for a,b in pairs:
        for ep,donor in [(a,b),(b,a)]:
            obs,anchor=hist[ep]
            end=endpoints(dict(obs,overhead=hist[donor][0]['overhead']),anchor)
            unit=goals[donor]-goals[ep]
            unit/=np.linalg.norm(unit)
            for mode,p in end.items():
                shift=p-original[ep][mode]
                directions.append(dict(recipient=ep,donor=donor,mode=mode,signed_mm=shift@unit,
                    amplitude_mm=np.linalg.norm(shift)))
    hook.remove()
    audit=json.loads((out/'audit_inputs.json').read_bytes())
    audit['phase_matched']=dict(ticks=ticks,pairs=pairs,natural=natural,directions=directions,
        policy_sha256=fit_record['policy_sha256'])
    rel.dump(out/'audit_inputs.json',audit)
    print('phase-matched pairs from original 16:',len(pairs),flush=True)


def verify_selection(work,out):
    """Rescan the training feedback as arrays and verify all qualifying start indices."""
    xy,_,_=setup(work)
    rel,_,records,geo,_,fk,_,active,_,_,_=xy.load_windows(work)
    selection=json.loads((out/'selection.json').read_bytes())
    count=0
    max_difference=0.
    for ep in sorted(active):
        r=records[ep]
        group=selection['episodes'][ep]
        center=np.r_[r['uv'],1.]@np.array(geo['table_map'])
        tcp=fk(r['q'])
        contacts=tcp[r['contact']]
        end=min(k for k in (r['close'],r['contact']) if k is not None)
        ticks=np.arange(7,end-8)
        src=ticks[:,None]+np.arange(9)
        pos=tcp[src]
        heights=xy.pose(r['q'][:,:6],np.asarray(geo['tip_mm']))[0][:,2]
        times=np.asarray([json.loads(l)['feedback']['sample_monotonic_ns'] for l in
            (r['folder']/'alignment.jsonl').read_bytes().splitlines()[r['first']:]])
        d0=np.linalg.norm(pos[:,0]-center,axis=1)
        gain=d0-np.linalg.norm(pos[:,-1]-center,axis=1)
        second=np.linalg.norm(pos[:,0]-contacts,axis=1)-np.linalg.norm(pos[:,-1]-contacts,axis=1)
        arc=np.linalg.norm(np.diff(pos,axis=1),axis=2).sum(axis=1)
        good=((np.diff(times[src],axis=1)>0).all(axis=1)&(d0<=60)&(gain>=5)&(second>=5)&(gain>=.6*arc)
              &(np.linalg.norm(pos[:,-1]-pos[:,0],axis=1)>=np.ptp(heights[src],axis=1)))
        assert group['scanned']==len(ticks)
        assert ticks[good].tolist()==[v['t'] for v in group['qualifying']]
        recorded=np.array([v['advance_mm'] for v in group['qualifying']])
        if len(recorded):
            max_difference=max(max_difference,float(np.max(np.abs(gain[good]-recorded))))
        # fk() promotes archived float32 joints before converting to degrees; pose() promotes after.
        np.testing.assert_allclose(gain[good],recorded,rtol=0,atol=1e-4)
        count+=len(ticks)
    audit=json.loads((out/'audit_inputs.json').read_bytes())
    audit['independent_training_window_rescan']=count
    audit['rescan_max_advance_difference_mm']=max_difference
    rel.dump(out/'audit_inputs.json',audit)
    print('verified all qualifying indices across',count,'windows',flush=True)


if __name__=='__main__':
    work,out=map(Path,sys.argv[1:3])
    if sys.argv[3]=='select':
        select(work,out)
    elif sys.argv[3]=='fit':
        fit(work,out)
    elif sys.argv[3]=='score':
        xy,_,_=setup(work)
        xy.score(work,out)
        evaluation_context(work,out)
    elif sys.argv[3]=='replay':
        xy,_,_=setup(work)
        xy.replay(work,out)
    elif sys.argv[3]=='phase':
        phase_check(work,out)
    elif sys.argv[3]=='verify_selection':
        verify_selection(work,out)
