"""Offline policy supervised by future, measured planar TCP displacements in B1."""
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[variable]='2'
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def setup(work):
    prior=work/'offline_diagnostics/b1_relative_policy_20261125'
    sys.path.insert(0,str(prior))
    import experiment as relative
    relative.paths(work)
    return relative,prior


def load_windows(work):
    rel,prior=setup(work)
    records,geometry,goals,xy=rel.data(work)
    audit=json.loads((prior/'data_audit.json').read_bytes())
    assert rel.digest(prior/'private_windows.npz')==audit['private_archive_sha256']
    with np.load(prior/'private_windows.npz') as archive:
        d={k:archive[k] for k in archive.files}
    rec={r['episode']:r for r in records}
    active={ep for ep,r in rec.items() if r['train']}
    assert len(active)==50 and active.isdisjoint(goals)
    stamps={}
    for ep in set(d['episodes']):
        r=rec[ep]
        rows=[json.loads(l) for l in (r['folder']/'alignment.jsonl').read_bytes().splitlines()][r['first']:]
        stamps[ep]=np.array([a['feedback']['sample_monotonic_ns'] for a in rows],dtype='int64')
    target=[]
    start=[]
    spans=[]
    for ep,t in zip(d['episodes'],d['indices']):
        q=rec[ep]['q'][t:t+9]
        times=stamps[ep][t:t+9]
        assert len(q)==len(times)==9 and np.all(np.diff(times)>0)
        tr=xy(q)
        np.testing.assert_allclose(tr[0],xy(q[0]),atol=1e-10)
        start.append(tr[0])
        target.append(tr[1:]-tr[0])
        spans.append((times[-1]-times[0])/1e6)
    return rel,prior,rec,geometry,goals,xy,d,active,np.array(start),np.array(target),np.array(spans)


def fit(work,out):
    rel,prior,rec,geo,goals,xy,d,active,start,target,spans=load_windows(work)
    _,coord=rel.paths(work)
    eps=d['episodes']
    tr=np.flatnonzero(np.isin(eps,sorted(active)))
    te=np.flatnonzero(~np.isin(eps,sorted(active)))
    assert len(tr)==200 and len(te)==33
    y=target.reshape(len(target),16)
    query=d['features'].astype(float)
    readers=rel.coordinate_heads(coord,query)
    memo={}
    def decode(groups):
        key=frozenset(groups)
        if key not in memo:
            memo[key]=np.stack([r.fit(key)[0] for r in readers],axis=1)
        return memo[key]
    def crossfit(groups):
        ans=np.full((len(eps),2),np.nan)
        for val in rel.folds(groups):
            sel=np.isin(eps,sorted(val))
            wood,jaw=decode(groups-val)[sel].transpose(1,0,2)
            ans[sel]=wood-jaw
        return ans
    loss={k:np.zeros(len(rel.PENALTIES)) for k in ('xy','permuted')}
    constant_loss=0.
    for val in rel.folds(active):
        groups=active-val
        learn=np.flatnonzero(np.isin(eps,sorted(groups)))
        valid=np.flatnonzero(np.isin(eps,sorted(val)))
        xx=crossfit(groups)[learn]
        v=decode(groups)[valid]
        xv=v[:,0]-v[:,1]
        w=rel.weights(eps[valid])
        wt=rel.weights(eps[learn])
        center=wt@y[learn]/sum(wt)
        def error(pred):
            return w@np.linalg.norm((pred-y[valid]).reshape(-1,8,2),axis=2).mean(1)
        constant_loss+=error(center)
        for mode,x in [('xy',xx),('permuted',rel.shuffle_by_demo(xx,eps[learn],d['offsets'][learn]))]:
            for i,p in enumerate(rel.PENALTIES):
                head=rel.fit_head(x,y[learn],eps[learn],p)
                loss[mode][i]+=error(rel.residual(head,xv))
    xx=crossfit(active)[tr]
    params={}
    fitted={}
    for mode,x in [('xy',xx),('permuted',rel.shuffle_by_demo(xx,eps[tr],d['offsets'][tr]))]:
        reg=rel.PENALTIES[int(np.argmin(loss[mode]))]
        head=rel.fit_head(x,y[tr],eps[tr],reg)
        params.update({mode+'_'+k:v for k,v in head.items()})
        fitted[mode]=reg
    for reader,key in zip(readers,['wood','jaw']):
        pred,w,b,_=reader.fit(active)
        np.testing.assert_allclose(query@w+b,pred,atol=1e-5)
        params[key+'_w'],params[key+'_b']=w,b
    np.savez(out/'private_xy_policy.npz',**params)
    provenance=json.loads((prior/'data_audit.json').read_bytes())
    rel.dump(out/'audit_inputs.json',dict(protocol_sha256=rel.digest(out/'protocol.md'),
        checkpoint_sha256=provenance['checkpoint_sha256'],windows_sha256=provenance['private_archive_sha256'],
        pairs=geo['pairs'],training_episodes=sorted(active),heldout_episodes=geo['test_episodes'],
        n_train=len(tr),n_held=len(te),coordinate_training=[len(r.groups) for r in readers],
        train_per_offset={str(k):int(sum(d['offsets'][tr]==k)) for k in rel.OFFSETS},
        feedback_span_ms_min_med_max=np.percentile(spans,[0,50,100]).tolist(),
        target_index='measured feedback t+1 through t+8 minus measured feedback t'))
    rel.dump(out/'fit.json',dict(candidate_penalties=rel.PENALTIES,selected_penalty=fitted,
        cv_mean_path_error_mm={k:(v/len(active)).tolist() for k,v in loss.items()},
        constant_cv_mean_path_error_mm=constant_loss/len(active),
        median_measured_endpoint_travel_mm={k:float(np.median(np.linalg.norm(target[s,-1],axis=1)))
            for k,s in [('train',tr),('heldout',te)]},policy_sha256=rel.digest(out/'private_xy_policy.npz')))
    print('fit on',len(tr),'feedback windows. Eight feedback intervals in ms:',np.percentile(spans,[0,50,100]),flush=True)


def pose(q,tip):
    from fit_gripper import flange
    matrix=flange(np.rad2deg(q))
    rot=matrix[...,:3,:3]
    xyz=matrix[...,:3,3]+rot@tip
    return xyz,rot


def joint_commands(base,desired_xy,tip):
    """Offline pose reconstruction, seeded at each original B1 joint command."""
    q0=np.asarray(base[...,:6],dtype=float).reshape(-1,6)
    xyz,rot=pose(q0,tip)
    goal=xyz.copy()
    goal[:,:2]=desired_xy.reshape(-1,2)
    def deviation(q,goal,rot):
        pos,orient=pose(q,tip)
        diff=rot.swapaxes(-1,-2)@orient
        rv=Rotation.from_matrix(diff.reshape(-1,3,3)).as_rotvec().reshape(pos.shape)
        return np.concatenate([pos-goal,100.*rv],axis=-1)
    q=q0.copy()
    step=1e-6
    for _ in range(18):
        residual=deviation(q,goal,rot)
        which=np.flatnonzero(np.linalg.norm(residual,axis=1)>1e-7)
        if not len(which):break
        z=q[which]
        variations=z[:,None,:]+step*np.eye(6)[None,:,:]
        r=deviation(variations,goal[which,None],rot[which,None])
        jac=((r-residual[which,None])/step).swapaxes(1,2)
        jt=jac.swapaxes(1,2)
        dq=np.linalg.solve(jt@jac+1e-5*np.eye(6),jt@residual[which,:,None])[...,0]
        dq/=np.maximum(1.,np.linalg.norm(dq,axis=1,keepdims=True)/0.15)
        q[which]-=dq
    rem=np.flatnonzero(np.linalg.norm(deviation(q,goal,rot),axis=1)>1e-5)
    for i in rem:
        solved=least_squares(lambda v:deviation(v[None],goal[i:i+1],rot[i:i+1])[0],q[i],
                             ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=100)
        q[i]=solved.x
    err=deviation(q,goal,rot)
    result=np.array(base,dtype=float,copy=True)
    result[...,:6]=q.reshape(base.shape[:-1]+(6,))
    assert np.array_equal(result[...,6],base[...,6])
    return result,dict(xy_max_mm=float(np.linalg.norm(err[:,:2],axis=1).max()),
        height_max_mm=float(abs(err[:,2]).max()),angle_max_deg=float(np.rad2deg(np.linalg.norm(err[:,3:],axis=1).max()/100)),
        joint_change_max_deg=float(np.rad2deg(abs(q-q0).max())),poses=len(q))


def decoded(params,feature):
    return [feature@params[k+'_w']+params[k+'_b'] for k in ('wood','jaw')]


def displacement(params,mode,wood,jaw):
    prefix='permuted_' if mode=='permuted' else 'xy_'
    pos=wood-jaw
    mean=params[prefix+'cy']
    result=mean if mode=='fixed' else mean+((pos-params[prefix+'cx'])/params[prefix+'sx'])@params[prefix+'slope']
    return result.reshape(8,2)


def score(work,out):
    rel,prior,rec,geo,goals,xy,d,active,start,target,_=load_windows(work)
    fit_record=json.loads((out/'fit.json').read_bytes())
    assert rel.digest(out/'private_xy_policy.npz')==fit_record['policy_sha256']
    with np.load(out/'private_xy_policy.npz') as archive:
        params={k:archive[k] for k in archive.files}
    held={e:i for i,e in enumerate(d['episodes']) if e not in active}
    tip=np.asarray(geo['tip_mm'])
    wood,jaw=decoded(params,d['features'].astype(float))
    originals={}
    swaps={}
    bases=[]
    targets=[]
    names=[]
    modes=('xy','fixed','permuted')
    for ep,i in held.items():
        originals[ep,'base']=xy(d['base'][i,-1])
        for mode in modes:
            names.append(('orig',ep,mode))
            bases.append(d['base'][i])
            targets.append(start[i]+displacement(params,mode,wood[i],jaw[i]))
    for k,(recipient,donor) in enumerate(zip(d['recipients'],d['donors'])):
        i,j=held[recipient],held[donor]
        swaps[recipient,'base']=xy(d['swapped'][k,-1])
        for mode in (*modes,'wood_only','jaw_only'):
            changed=mode in modes
            names.append(('swap',recipient,mode))
            w=wood[i] if mode=='jaw_only' else wood[j]
            g=jaw[i] if mode=='wood_only' else jaw[j]
            bases.append(d['swapped'][k] if changed else d['base'][i])
            targets.append(start[i]+displacement(params,mode,w,g))
    commands,ik=joint_commands(np.stack(bases),np.stack(targets),tip)
    for (kind,ep,mode),q in zip(names,commands):
        dest=originals if kind=='orig' else swaps
        dest[ep,mode]=xy(q[-1])
    natural=[]
    for ep,i in held.items():
        measured=start[i]+target[i,-1]
        for mode in ('base',*modes,'measured'):
            endpoint=measured if mode=='measured' else originals[ep,mode]
            natural.append(dict(episode=ep,mode=mode,feedback_endpoint_error_mm=float(np.linalg.norm(endpoint-measured)),
                distance_to_block_mm=float(np.linalg.norm(endpoint-goals[ep])),
                advance_mm=float(np.linalg.norm(start[i]-goals[ep])-np.linalg.norm(endpoint-goals[ep])),
                travel_mm=float(np.linalg.norm(endpoint-start[i]))))
    directions=[]
    for rec_ep,donor in zip(d['recipients'],d['donors']):
        truth=goals[donor]-goals[rec_ep]
        sep=np.linalg.norm(truth)
        axis=truth/sep
        side=np.array([-axis[1],axis[0]])
        contact=np.subtract(*[xy(rec[k]['q'][rec[k]['contact']]) for k in (donor,rec_ep)])
        contact/=np.linalg.norm(contact)
        for mode in ('base',*modes,'wood_only','jaw_only'):
            old=originals[rec_ep,mode if mode in ('base',*modes) else 'xy']
            new=swaps[rec_ep,mode]
            delta=new-old
            directions.append(dict(recipient=rec_ep,donor=donor,mode=mode,separation_mm=float(sep),
                signed_mm=float(delta@axis),orthogonal_mm=float(delta@side),amplitude_mm=float(np.linalg.norm(delta)),
                contact_signed_mm=float(delta@contact),donor_distance_reduction_mm=float(
                    np.linalg.norm(old-goals[donor])-np.linalg.norm(new-goals[donor]))))
    rel.dump(out/'outcomes.json',dict(natural=natural,directions=directions,pose_reconstruction=ik))
    print('scored',len(natural),'natural metrics and',len(directions),'directional metrics',ik,flush=True)


def measured_pairs(work,out):
    """Describe the paired demonstrators' natural movements, with their original starting poses."""
    rel,prior,rec,geo,goals,xy,d,active,start,target,_=load_windows(work)
    held={ep:i for i,ep in enumerate(d['episodes']) if ep not in active}
    items=[]
    for a,b in geo['pairs']:
        direction=goals[b]-goals[a]
        unit=direction/np.linalg.norm(direction)
        delta=target[held[b],-1]-target[held[a],-1]
        items.append(dict(pair=[a,b],measured_displacement_difference_projected_mm=float(delta@unit),
            original_start_separation_mm=float(np.linalg.norm(start[held[b]]-start[held[a]]))))
    record=json.loads((out/'audit_inputs.json').read_bytes())
    record['natural_expert_pairs']=items
    rel.dump(out/'audit_inputs.json',record)
    print('checked measured natural displacement in',len(items),'pairs')


def replay(work,out):
    rel,prior,rec,geo,goals,xy,d,active,start,target,_=load_windows(work)
    import cv2
    import torch
    import torch.nn.functional as F
    from probe import load_model
    from annotate import images
    from train_contrast import image_rgb
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    net,sha=load_model(work,'baseline')
    assert sha==json.loads((out/'audit_inputs.json').read_bytes())['checkpoint_sha256']
    fit_record=json.loads((out/'fit.json').read_bytes())
    assert rel.digest(out/'private_xy_policy.npz')==fit_record['policy_sha256']
    with np.load(out/'private_xy_policy.npz') as archive:
        params={k:archive[k] for k in archive.files}
    features=[]
    def extract(module,inputs,activation):
        features.append(F.adaptive_avg_pool2d(activation[-1:],(8,8)).flatten().numpy().astype(float))
    handle=net.obs_encoder.key_model_map['overhead'].layer2.register_forward_hook(extract)
    tip=np.asarray(geo['tip_mm'])
    def predict_endpoint(obs,anchor):
        features.clear()
        with torch.inference_mode():
            baseline=net.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].numpy()
        assert len(features)==1
        wood,jaw=decoded(params,features[0])
        q0=obs['agent_pos'][0,-1].numpy()
        request=xy(q0)+displacement(params,'xy',wood,jaw)
        commands,_=joint_commands(baseline,request,tip)
        return xy(commands[-1])
    result=json.loads((out/'outcomes.json').read_bytes())
    natural={r['episode']:r for r in result['natural'] if r['mode']=='xy'}
    endpoints={}
    histories={}
    errors=[]
    for ep,row in natural.items():
        r=rec[ep]
        j=r['close']-18
        obs={'agent_pos':torch.from_numpy(r['q'][None,j-7:j+1])}
        for view in ('overhead','wrist'):
            files=images(r['folder'],view)[r['first']+j-7:r['first']+j+1]
            pixels=np.stack([image_rgb(cv2.imread(str(f)),128) for f in files])
            obs[view]=torch.from_numpy(pixels[None]).float()/255
        anchor=torch.from_numpy(r['cmd'][None,j-1])
        histories[ep]=(obs,anchor)
        p=predict_endpoint(obs,anchor)
        endpoints[ep]=p
        errors.append(abs(np.linalg.norm(p-xy(r['q'][j+8]))-row['feedback_endpoint_error_mm']))
        errors.append(abs(np.linalg.norm(p-goals[ep])-row['distance_to_block_mm']))
    rows=[r for r in result['directions'] if r['mode']=='xy']
    for r in rows:
        a,b=r['recipient'],r['donor']
        obs,anchor=histories[a]
        p=predict_endpoint(dict(obs,overhead=histories[b][0]['overhead']),anchor)
        axis=goals[b]-goals[a]
        delta=p-endpoints[a]
        errors.append(abs(delta@axis/np.linalg.norm(axis)-r['signed_mm']))
        errors.append(abs(np.linalg.norm(delta)-r['amplitude_mm']))
    handle.remove()
    assert max(errors)<1e-5
    fit_record['image_to_joint_verification']=dict(natural=len(natural),swapped=len(rows),max_scalar_difference_mm=max(errors))
    rel.dump(out/'fit.json',fit_record)
    print('image-to-joint replay',fit_record['image_to_joint_verification'],flush=True)


if __name__=='__main__':
    work,out=map(Path,sys.argv[1:3])
    if sys.argv[3]=='fit':
        fit(work,out)
    elif sys.argv[3]=='score':
        score(work,out)
    elif sys.argv[3]=='replay':
        replay(work,out)
    elif sys.argv[3]=='measured':
        measured_pairs(work,out)
