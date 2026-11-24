"""Offline B1 policy with a learned residual conditioned on the decoded block-to-jaw offset."""
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='2'
import cv2
import numpy as np
import torch
import torch.nn.functional as F

OFFSETS=(30,24,18,12)
PENALTIES=(0.01,0.1,1.,10.,100.)
SEED=1125


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,indent=1)+'\n')
    tmp.replace(path)


def paths(work):
    diag=work/'offline_diagnostics'
    prior=diag/'b1_vision_contrasts_20261124'
    coord=diag/'b1_coordinate_probe_20261125'
    sys.path[:0]=[str(coord),str(prior)]
    return prior,coord


def data(work):
    prior,coord=paths(work)
    from assess import geometry,reviewed
    from train_contrast import MANIFEST
    sel=work/'offline_diagnostics/b1_ode_lerobot_20261123/selection.json'
    assert digest(sel)==MANIFEST
    selected=set(json.loads(sel.read_bytes())['selected_ids'])
    geo,test,goals,xy=geometry(work,prior)
    assert geo['pairs']==json.loads((prior/'final.json').read_bytes())['geometry']['pairs']
    records=reviewed(work,prior)
    assert {r['episode'] for r in records if r['train']}==selected and len(selected)==50
    assert selected.isdisjoint(geo['test_episodes'])
    return records,geo,goals,xy


@torch.inference_mode()
def prepare(work,out):
    records,geo,_,_=data(work)
    from probe import load_model
    from annotate import images
    from train_contrast import image_rgb
    model,sha=load_model(work,'baseline')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    mid=[]
    def capture(module,inputs,outputs):
        mid.append(F.adaptive_avg_pool2d(outputs[-1:],(8,8)).flatten().cpu().numpy())
    hook=model.obs_encoder.key_model_map['overhead'].layer2.register_forward_hook(capture)
    def base(obs,anchor):
        mid.clear()
        commands=model.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].numpy()
        assert len(mid)==1 and commands.shape==(8,7)
        return commands.copy(),mid[0].copy()
    episodes=[]
    offsets=[]
    indices=[]
    features=[]
    predictions=[]
    expert=[]
    held={}
    for r in records:
        ep=r['episode']
        if not r['train'] and ep not in geo['test_episodes']:
            continue
        near=r['close'] if r['close'] is not None else r['contact']
        if near is None:continue
        files={view:images(r['folder'],view)[r['first']:] for view in ('overhead','wrist')}
        for offset in OFFSETS if r['train'] else (18,):
            j=near-offset
            if j<7 or j+7>=len(r['cmd']):continue
            obs={'agent_pos':torch.from_numpy(r['q'][None,j-7:j+1])}
            for view,fs in files.items():
                arr=np.stack([image_rgb(cv2.imread(str(f)),128) for f in fs[j-7:j+1]])
                obs[view]=torch.from_numpy(arr[None]).float()/255
            anchor=torch.from_numpy(r['cmd'][None,j-1])
            pred,feat=base(obs,anchor)
            episodes.append(ep)
            offsets.append(offset)
            indices.append(j)
            features.append(feat)
            predictions.append(pred)
            expert.append(r['cmd'][j:j+8])
            if not r['train']:held[ep]=(obs,anchor,feat)
        if len(episodes)%25<4: print('natural archive windows',len(episodes),flush=True)
    assert set(held)==set(geo['test_episodes'])
    donors=[]
    recipients=[]
    swapped=[]
    for a,b in geo['pairs']:
        for ep,other in ((a,b),(b,a)):
            obs,anchor,_=held[ep]
            mod=dict(obs,overhead=held[other][0]['overhead'])
            pred,feat=base(mod,anchor)
            np.testing.assert_allclose(feat,held[other][2],atol=1e-6)
            recipients.append(ep)
            donors.append(other)
            swapped.append(pred)
    hook.remove()
    prior,coord=paths(work)
    index=json.loads((coord/'frames.json').read_bytes())
    by_ep={r['episode']:r for r in index if r['phase']=='approach'}
    with np.load(coord/'baseline_features.npz') as cache:
        keys=cache['keys'].tolist()
        archive=cache['layer2_map']
        for ep,j,offset,ft in zip(episodes,indices,offsets,features):
            if offset==18:
                assert j==by_ep[ep]['j']
                np.testing.assert_allclose(ft,archive[keys.index(by_ep[ep]['key'])],rtol=5e-4,atol=3e-5)
    temp=out/'private_windows.partial.npz'
    np.savez(temp,episodes=episodes,offsets=offsets,indices=indices,features=features,base=predictions,
             expert=expert,donors=donors,recipients=recipients,swapped=swapped)
    temp.replace(out/'private_windows.npz')
    dump(out/'data_audit.json',dict(checkpoint_sha256=sha,protocol_sha256=digest(out/'protocol.md'),
         windows=len(episodes),heldout_natural=len(held),paired_directions=len(swapped),
         train_episode_count=len(set(episodes)-set(held)),index_sha256=digest(coord/'frames.json'),
         prior_pairs=geo['pairs'],coordinate_labels_sha256=digest(coord/'labels.json'),
         private_archive_sha256=digest(out/'private_windows.npz')))
    print('prepared',len(episodes),'windows and',len(swapped),'paired directions',flush=True)


class Coordinate:
    """Training-only ridge decoder. Recenter each kernel on its own training subset."""
    def __init__(self,x,y,groups,query):
        self.mean=x.mean(0)
        self.x=x-self.mean
        self.y=y
        self.groups=np.asarray(groups)
        self.gram=self.x@self.x.T
        self.cross=(query-self.mean)@self.x.T

    def solve(self,train,cross,penalties):
        k=self.gram[np.ix_(train,train)]
        cross=cross[:,train]
        km=k.mean(0)
        k=k-km-km[:,None]+km.mean()
        cross=cross-km-cross.mean(1,keepdims=True)+km.mean()
        scale=max(float(k.diagonal().mean()),1e-20)
        eig,vec=np.linalg.eigh(k/scale)
        left=cross@vec/scale
        yc=self.y[train].mean(0)
        right=vec.T@(self.y[train]-yc)
        for p in penalties:
            coeff=right/(np.maximum(eig,0)[:,None]+p)
            yield yc+left@coeff,vec@coeff/scale

    def fit(self,allowed):
        train=np.flatnonzero(np.isin(self.groups,list(allowed)))
        penalties=np.logspace(-5,1,7)
        losses=np.zeros(len(penalties))
        for val in np.array_split(np.random.default_rng(317).permutation(train),5):
            learn=np.setdiff1d(train,val)
            for i,(p,_) in enumerate(self.solve(learn,self.gram[val],penalties)):
                losses[i]+=np.linalg.norm(p-self.y[val],axis=1).sum()
        alpha=penalties[np.argmin(losses)]
        pred,dual=next(self.solve(train,self.cross,[alpha]))
        w=(self.x[train]-self.x[train].mean(0)).T@dual
        b=self.y[train].mean(0)-(self.mean+self.x[train].mean(0))@w
        return pred,w,b,float(alpha)


def folds(groups):
    return [set(f.tolist()) for f in np.array_split(np.random.default_rng(SEED).permutation(sorted(groups)),5)]


def weights(eps):
    _,inverse,counts=np.unique(eps,return_inverse=True,return_counts=True)
    return 1./counts[inverse]


def fit_head(x,y,eps,penalty):
    sample=weights(eps)
    sample/=sample.sum()
    cx=sample@x
    sx=np.sqrt(sample@((x-cx)**2)).clip(1e-6)
    z=(x-cx)/sx
    cy=sample@y
    # Penalty is expressed in effective demonstration units.
    cov=z.T@(sample[:,None]*z)
    slope=np.linalg.solve(cov+np.eye(2)*penalty/len(np.unique(eps)),z.T@(sample[:,None]*(y-cy)))
    return dict(cx=cx,sx=sx,cy=cy,slope=slope)


def residual(head,x):
    return head['cy']+((x-head['cx'])/head['sx'])@head['slope']


def corrected(base,resid):
    result=base.astype(np.float64).copy()
    result[...,:6]+=resid.reshape(8,6)
    assert np.array_equal(result[:,6],base[:,6])
    return result


def shuffle_by_demo(x,eps,offsets):
    names=np.random.default_rng(SEED).permutation(np.unique(eps))
    mapping=dict(zip(names,np.roll(names,1)))
    result=[]
    for ep,t in zip(eps,offsets):
        choices=np.flatnonzero(eps==mapping[ep])
        best=choices[np.argmin(abs(offsets[choices]-t))]
        result.append(x[best])
    return np.stack(result)


def coordinate_heads(coord,features):
    rows=json.loads((coord/'frames.json').read_bytes())
    jaws=json.loads((coord/'labels.json').read_bytes())['jaw']
    with np.load(coord/'baseline_features.npz') as archive:
        assert archive['keys'].tolist()==[r['key'] for r in rows]
        labeled=archive['layer2_map'].astype(float)
    heads=[]
    for phase in ('review','approach'):
        use=[i for i,r in enumerate(rows) if r['train'] and r['phase']==phase and
             (phase=='review' or r['key'] in jaws)]
        y=[rows[i]['uv'] if phase=='review' else np.mean(jaws[rows[i]['key']],axis=0) for i in use]
        heads.append(Coordinate(labeled[use],np.array(y),[rows[i]['episode'] for i in use],features))
    return heads


def fit_policy(work,out):
    records,geo,goals,xy=data(work)
    prior,coord=paths(work)
    checks=json.loads((out/'data_audit.json').read_bytes())
    assert digest(out/'private_windows.npz')==checks['private_archive_sha256']
    active={r['episode'] for r in records if r['train']}
    with np.load(out/'private_windows.npz') as archive:
        d={k:archive[k] for k in archive.files}
    eps=d['episodes']
    train=np.flatnonzero(np.isin(eps,sorted(active)))
    held=np.flatnonzero(~np.isin(eps,sorted(active)))
    assert set(eps[held])==set(geo['test_episodes'])
    query=d['features'].astype(float)
    readers=coordinate_heads(coord,query)
    fit_cache={}
    def decode(groups):
        key=frozenset(groups)
        if key not in fit_cache:
            fit_cache[key]=np.stack([r.fit(key)[0] for r in readers],axis=1)
        return fit_cache[key]
    def delta(coords):
        return coords[:,0]-coords[:,1]
    def crossfit(groups):
        result=np.full((len(eps),2),np.nan)
        for val in folds(groups):
            subset=np.isin(eps,sorted(val))
            result[subset]=delta(decode(groups-val)[subset])
        return result
    target=(d['expert'][:,:,:6].astype(float)-d['base'][:,:,:6]).reshape(len(eps),48)
    losses={k:np.zeros(len(PENALTIES)) for k in ('relative','permuted')}
    for val_groups in folds(active):
        groups=active-val_groups
        t=np.flatnonzero(np.isin(eps,sorted(groups)))
        v=np.flatnonzero(np.isin(eps,sorted(val_groups)))
        xx=crossfit(groups)[t]
        xv=delta(decode(groups)[v])
        perm=shuffle_by_demo(xx,eps[t],d['offsets'][t])
        for name,x in [('relative',xx),('permuted',perm)]:
            for i,reg in enumerate(PENALTIES):
                h=fit_head(x,target[t],eps[t],reg)
                errors=abs(residual(h,xv)-target[v]).mean(1)
                losses[name][i]+=weights(eps[v])@errors
    xx=crossfit(active)[train]
    all_coords=decode(active)
    heads={}
    alphas={}
    for name,x in [('relative',xx),('permuted',shuffle_by_demo(xx,eps[train],d['offsets'][train]))]:
        alpha=PENALTIES[np.argmin(losses[name])]
        alphas[name]=alpha
        heads[name]=fit_head(x,target[train],eps[train],alpha)
    params={}
    readout_alphas={}
    for reader,k in zip(readers,['wood','jaw']):
        pred,w,b,alpha=reader.fit(active)
        np.testing.assert_allclose(query@w+b,pred,atol=1e-5)
        params[k+'_w'],params[k+'_b']=w,b
        readout_alphas[k]=alpha
    for k,v in heads['relative'].items():params['action_'+k]=v
    np.savez(out/'private_relative_policy.npz',**params)
    held_by_ep={eps[i]:i for i in held}
    assert len(held_by_ep)==len(held)
    split={r['episode']:r for r in records}
    def rollout(i,base,mode,wood,jaw):
        if mode=='base':return base
        if mode=='fixed':return corrected(base,heads['relative']['cy'])
        h=heads['permuted'] if mode=='permuted' else heads['relative']
        return corrected(base,residual(h,wood-jaw))
    natural=[]
    base_modes=('base','fixed','permuted','relative')
    for ep,i in held_by_ep.items():
        r=split[ep]
        j=d['indices'][i]
        start=xy(r['cmd'][j-1])
        teacher=d['expert'][i]
        for mode in (*base_modes,'expert'):
            pred=teacher if mode=='expert' else rollout(i,d['base'][i],mode,*all_coords[i])
            endpoint=xy(pred[-1])
            natural.append(dict(episode=ep,mode=mode,
                initial_distance_mm=float(np.linalg.norm(goals[ep]-start)),
                end_distance_mm=float(np.linalg.norm(goals[ep]-endpoint)),
                endpoint_error_vs_expert_mm=float(np.linalg.norm(xy(teacher[-1])-endpoint)),
                arm_mae_deg=float(np.rad2deg(abs(pred[:,:6]-teacher[:,:6]).mean()))))
    directions=[]
    for k,(recipient,donor) in enumerate(zip(d['recipients'],d['donors'])):
        i,j=held_by_ep[recipient],held_by_ep[donor]
        truth=goals[donor]-goals[recipient]
        size=np.linalg.norm(truth)
        axis=truth/size
        tangent=np.array([-axis[1],axis[0]])
        contact=np.subtract(*[xy(split[name]['q'][split[name]['contact']]) for name in [donor,recipient]])
        contact=contact/max(1e-8,np.linalg.norm(contact))
        for mode in (*base_modes,'wood_only','jaw_only'):
            original=rollout(i,d['base'][i],mode,*all_coords[i])
            wood,jaw=all_coords[j]
            swapped=d['swapped'][k]
            if mode=='wood_only':jaw,swapped=all_coords[i,1],d['base'][i]
            if mode=='jaw_only':wood,swapped=all_coords[i,0],d['base'][i]
            changed=rollout(i,swapped,mode,wood,jaw)
            old_xy=xy(original[-1])
            new_xy=xy(changed[-1])
            shift=new_xy-old_xy
            directions.append(dict(recipient=recipient,donor=donor,mode=mode,separation_mm=float(size),
                signed_mm=float(shift@axis),orthogonal_mm=float(shift@tangent),
                amplitude_mm=float(np.linalg.norm(shift)),contact_signed_mm=float(shift@contact),
                donor_distance_reduction_mm=float(np.linalg.norm(goals[donor]-old_xy)-np.linalg.norm(goals[donor]-new_xy))))
    dump(out/'fit.json',dict(coordinate_training={k:len(v.groups) for k,v in zip(('wood','jaw'),readers)},
         readout_alphas=readout_alphas,head_alphas=alphas,penalties=PENALTIES,
         cv_mae_deg={k:np.rad2deg(v/len(set(eps[train]))).tolist() for k,v in losses.items()},
         train_episodes=sorted(set(eps[train])),train_window_counts={str(t):int(sum(d['offsets'][train]==t)) for t in OFFSETS},
         private_policy_sha256=digest(out/'private_relative_policy.npz')))
    dump(out/'outcomes.json',dict(directions=directions,natural=natural))
    print('finished',len(directions),'interventions and',len(natural),'natural metrics',flush=True)


@torch.inference_mode()
def predict_with_relative(model,obs,anchor,params):
    """Image-to-joint offline entry point, using only images, feedback, previous command and fitted weights."""
    spatial=[]
    def observe(module,inputs,activation):
        spatial.append(F.adaptive_avg_pool2d(activation[-1:],(8,8)).flatten().numpy().astype(float))
    hook=model.obs_encoder.key_model_map['overhead'].layer2.register_forward_hook(observe)
    try:
        base=model.predict_action(obs,a0=anchor,eps=torch.zeros_like(anchor))['action'][0].numpy()
    finally:
        hook.remove()
    assert len(spatial)==1
    locations=[spatial[0]@params[key+'_w']+params[key+'_b'] for key in ('wood','jaw')]
    h={k:params['action_'+k] for k in ('cx','sx','cy','slope')}
    return corrected(base,residual(h,locations[0]-locations[1]))


def replay(work,out):
    records,geo,goals,xy=data(work)
    from probe import load_model
    from annotate import images
    from train_contrast import image_rgb
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model,identity=load_model(work,'baseline')
    audit=json.loads((out/'data_audit.json').read_bytes())
    fitted=json.loads((out/'fit.json').read_bytes())
    assert identity==audit['checkpoint_sha256'] and digest(out/'private_relative_policy.npz')==fitted['private_policy_sha256']
    with np.load(out/'private_relative_policy.npz') as archive:
        params={k:archive[k] for k in archive.files}
    results=json.loads((out/'outcomes.json').read_bytes())
    natural={r['episode']:r for r in results['natural'] if r['mode']=='relative'}
    directions={(r['recipient'],r['donor']):r for r in results['directions'] if r['mode']=='relative'}
    hist={}
    endpoints={}
    errors=[]
    for r in records:
        ep=r['episode']
        if ep not in natural:continue
        j=r['close']-18
        obs={'agent_pos':torch.from_numpy(r['q'][None,j-7:j+1])}
        for role in ('overhead','wrist'):
            ff=images(r['folder'],role)[r['first']+j-7:r['first']+j+1]
            rgb=np.stack([image_rgb(cv2.imread(str(f)),128) for f in ff])
            obs[role]=torch.from_numpy(rgb[None]).float()/255
        anchor=torch.from_numpy(r['cmd'][None,j-1])
        hist[ep]=(obs,anchor)
        end=xy(predict_with_relative(model,obs,anchor,params)[-1])
        errors.append(abs(np.linalg.norm(end-goals[ep])-natural[ep]['end_distance_mm']))
        endpoints[ep]=end
    for (rec,donor),row in directions.items():
        obs,anchor=hist[rec]
        changed=dict(obs,overhead=hist[donor][0]['overhead'])
        shift=xy(predict_with_relative(model,changed,anchor,params)[-1])-endpoints[rec]
        axis=goals[donor]-goals[rec]
        errors.append(abs(shift@axis/np.linalg.norm(axis)-row['signed_mm']))
        errors.append(abs(np.linalg.norm(shift)-row['amplitude_mm']))
    assert max(errors)<1e-6
    with np.load(out/'private_windows.npz') as d:
        t=np.flatnonzero(np.isin(d['episodes'],fitted['train_episodes']))
        e=d['episodes'][t]
        y=(d['expert'][t,:,:6].astype(float)-d['base'][t,:,:6]).reshape(len(t),48)
        bias_error=0.
        for val in folds(set(e.tolist())):
            v=np.isin(e,list(val))
            wt=weights(e[~v])
            offset=wt@y[~v]/sum(wt)
            bias_error+=weights(e[v])@abs(y[v]-offset).mean(1)
        train_base_mae=weights(e)@abs(y).mean(1)/len(set(e))
    fitted['verification']=dict(natural_images_to_joints=len(hist),paired_image_swaps=len(directions),
        maximum_metric_difference_mm=max(errors),train_base_mae_deg=float(np.rad2deg(train_base_mae)),
        train_cv_bias_only_mae_deg=float(np.rad2deg(bias_error/len(set(e)))))
    dump(out/'fit.json',fitted)
    print('image-to-joint replay verified',fitted['verification'],flush=True)


if __name__=='__main__':
    work,out=map(Path,sys.argv[1:3])
    if sys.argv[3]=='prepare':
        prepare(work,out)
    elif sys.argv[3]=='fit':
        fit_policy(work,out)
    elif sys.argv[3]=='replay':
        replay(work,out)
