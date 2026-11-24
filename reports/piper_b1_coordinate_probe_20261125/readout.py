"""Train an auxiliary linear coordinate readout on frozen features, splitting by demonstration."""
import hashlib
import json
import os
import sys
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='2'
import numpy as np


ALPHAS=np.logspace(-5,1,7)


def predict_many(x,y,train,test,alphas):
    mean=x[train].mean(axis=0)
    train_x=x[train]-mean
    gram=train_x@train_x.T
    scale=max(np.mean(gram.diagonal()),1e-24)
    eig,vec=np.linalg.eigh(gram/scale)
    eig=np.maximum(eig,0.)
    cross=((x[test]-mean)@train_x.T/scale)@vec
    center=y[train].mean(axis=0)
    targets=vec.T@(y[train]-center)
    return [center+(cross/(eig+a))@targets for a in alphas]


def fit(x,y,train,test):
    # Exactly one frame per episode in each fit. Randomize training episodes only.
    splits=np.array_split(np.random.default_rng(317).permutation(train),5)
    scores=np.zeros(len(ALPHAS))
    for fold in splits:
        learn=np.setdiff1d(train,fold)
        for i,pred in enumerate(predict_many(x,y,learn,fold,ALPHAS)):
            scores[i]+=np.linalg.norm(pred-y[fold],axis=1).sum()
    alpha=float(ALPHAS[np.argmin(scores)])
    return predict_many(x,y,train,test,[alpha])[0],alpha


def metrics(y,pred,base):
    delta=pred-y
    err=np.linalg.norm(delta,axis=1)
    default=np.linalg.norm(base-y,axis=1)
    rng=np.random.default_rng(317)
    idx=rng.integers(len(err),size=(10000,len(err)))
    return dict(mean_px=float(err.mean()),median_px=float(np.median(err)),
                p90_px=float(np.percentile(err,90)),mae_uv_px=np.abs(delta).mean(0).tolist(),
                within_10px=int(sum(err<=10)),within_15px=int(sum(err<=15)),n=len(err),
                r2_vs_train_center=float(1-err.dot(err)/default.dot(default)),
                mean_improvement_95ci_px=np.percentile((default-err)[idx].mean(1),[2.5,97.5]).tolist())


def main(work,out):
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    sys.path.insert(0,str(prior))
    from assess import reviewed
    records={r['episode']:r for r in reviewed(work,prior)}
    frame_index=json.loads((out/'frames.json').read_bytes())
    label_bytes=(out/'labels.json').read_bytes().replace(b'\r\n',b'\n')
    labels=json.loads(label_bytes)['jaw']
    archive=json.loads((prior/'final.json').read_bytes())
    result=dict(checkpoint_ids=archive['checkpoint_ids'],label_sha256=hashlib.sha256(label_bytes).hexdigest(),scores={})
    detail={}
    for variant in ['baseline','high','shared']:
        with np.load(out/(variant+'_features.npz')) as features:
            assert [r['key'] for r in frame_index]==features['keys'].tolist()
            for phase in ['review','approach','release']:
                ix=[i for i,r in enumerate(frame_index) if r['phase']==phase and
                    (phase=='review' or r['key'] in labels)]
                rows=[frame_index[i] for i in ix]
                episodes=[r['episode'] for r in rows]
                assert len(set(episodes))==len(rows)
                y=np.array([r['uv'] if phase=='review' else np.mean(labels[r['key']],axis=0)
                            for r in rows],dtype=float)
                train=np.array([i for i,r in enumerate(rows) if r['train']])
                test=np.array([i for i,r in enumerate(rows) if not r['train']])
                center=np.broadcast_to(y[train].mean(0),y[test].shape)
                name=variant+'/'+('block' if phase=='review' else 'jaw_'+phase)
                summary=result['scores'][name]=dict(train=len(train),heldout=len(test),
                    heldout_span_uv_px=np.ptp(y[test],axis=0).tolist(),constant=metrics(y[test],center,center))
                examples=[dict(episode=episodes[t],key=rows[t]['key'],target=y[t].tolist()) for t in test]
                detail[name]=examples
                q=np.stack([records[r['episode']]['q'][r['j']] for r in rows]).astype(float)
                # Express arm angles and opening on commensurate scales for the nuisance comparison.
                q=(q-q[train].mean(0))/np.maximum(q[train].std(0),1e-6)
                for rep in ['state_only','pixels','layer2_map','layer4_map','pooled','shuffled_pooled']:
                    x=q if rep=='state_only' else features['pooled' if rep=='shuffled_pooled' else rep][ix].astype(float)
                    target=y.copy()
                    if rep=='shuffled_pooled':
                        target[train]=target[np.random.default_rng(271).permutation(train)]
                    pred,alpha=fit(x,target,train,test)
                    summary[rep]=dict(alpha=alpha,**metrics(y[test],pred,center))
                    for record,pr in zip(examples,pred):
                        record[rep]=pr.round(6).tolist()
        print(variant,'readouts complete',flush=True)
    for name,obj in [('readout_results',result),('heldout_predictions',detail)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=1,allow_nan=False)+'\n')

def regions(work,out):
    """Remove the same workspace rectangle in every frame, independent of held-out labels."""
    import zipfile
    import cv2
    import torch
    import torch.nn.functional as functional
    from probe import load_model
    model,_=load_model(work,'baseline')
    from train_contrast import image_rgb
    torch.set_num_threads(2)
    rows=[r for r in json.loads((out/'frames.json').read_bytes()) if r['phase']=='review']
    train=np.array([i for i,r in enumerate(rows) if r['train']])
    test=np.array([i for i,r in enumerate(rows) if not r['train']])
    y=np.array([r['uv'] for r in rows])
    x0,y0=np.floor(y[train].min(0)-30).astype(int)
    x1,y1=np.ceil(y[train].max(0)+31).astype(int)
    center=np.broadcast_to(y[train].mean(0),y[test].shape)
    report=dict(rectangle_xyxy=[int(v) for v in [x0,y0,x1,y1]],fill_rgb=[128]*3,scores={})
    with zipfile.ZipFile(out/'frames.zip') as source:
        imgs=[cv2.imdecode(np.frombuffer(source.read(r['key']+'.jpg'),np.uint8),1) for r in rows]
    net=model.obs_encoder.key_model_map['overhead']
    transform=model.obs_encoder.key_transform_map['overhead']
    for mode in ['intact','region_only','region_hidden']:
        batches={k:[] for k in ['pixels','layer2_map','layer4_map','pooled']}
        for i in range(0,len(rows),12):
            pixels=[]
            for im in imgs[i:i+12]:
                a=im.copy()
                if mode=='region_only':
                    a.fill(128)
                    a[y0:y1,x0:x1]=im[y0:y1,x0:x1]
                if mode=='region_hidden':
                    a[y0:y1,x0:x1]=128
                pixels.append(image_rgb(a,128))
            with torch.inference_mode():
                x=transform(torch.from_numpy(np.stack(pixels)).float()/255)
                l2=net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x))))))
                l4=net.layer4(net.layer3(l2))
                pooled=net.fc(net.avgpool(l4).flatten(1))
                for k,v in [('pixels',functional.adaptive_avg_pool2d(x,(24,24))),
                            ('layer2_map',functional.adaptive_avg_pool2d(l2,(8,8))),
                            ('layer4_map',l4),('pooled',pooled)]:
                    batches[k].append(v.flatten(1).numpy())
        scores=report['scores'][mode]={}
        for key,parts in batches.items():
            pred,alpha=fit(np.concatenate(parts).astype(float),y,train,test)
            scores[key]=dict(alpha=alpha,**metrics(y[test],pred,center))
    (out/'region_results.json').write_text(json.dumps(report,indent=1,allow_nan=False)+'\n')


def approach_transfer(work,out):
    """Freeze the review-frame wood readout and score placement at the action probe's anchor frame."""
    rows=json.loads((out/'frames.json').read_bytes())
    idx={r['key']:i for i,r in enumerate(rows)}
    review=[r for r in rows if r['phase']=='review']
    train_rows=[r for r in review if r['train']]
    held=[r for r in review if not r['train']]
    training=np.arange(len(train_rows))
    test=np.arange(len(train_rows),len(review))
    y=np.array([r['uv'] for r in train_rows+held])
    numbers=[idx[r['key']] for r in train_rows]+[idx[f"{r['index']:02d}_approach"] for r in held]
    prior=work/'offline_diagnostics/b1_vision_contrasts_20261124'
    pairs=json.loads((prior/'final.json').read_bytes())['geometry']['pairs']
    by_ep={r['episode']:i for i,r in enumerate(held)}
    report={}
    for variant in ('baseline','high','shared'):
        entry=report[variant]={}
        with np.load(out/(variant+'_features.npz')) as features:
            for key in ('pixels','layer2_map','layer4_map','pooled'):
                x=features[key][numbers].astype(float)
                pred,alpha=fit(x,y,training,test)
                signed=[]
                gains=[]
                for a,b in pairs:
                    i,j=by_ep[a],by_ep[b]
                    delta=y[test[j]]-y[test[i]]
                    length=np.linalg.norm(delta)
                    movement=(pred[j]-pred[i]).dot(delta)/length
                    signed.append(movement)
                    gains.append(movement/length)
                center=np.broadcast_to(y[training].mean(0),y[test].shape)
                entry[key]=dict(alpha=alpha,**metrics(y[test],pred,center),
                    positive_pairs=int(sum(v>0 for v in gains)),median_pair_gain=float(np.median(gains)),
                    median_pair_projected_px=float(np.median(signed)))
    (out/'approach_transfer.json').write_text(json.dumps(report,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    fn={'regions':regions,'approach':approach_transfer}.get(sys.argv[3] if len(sys.argv)>3 else '',main)
    fn(*map(Path,sys.argv[1:3]))
