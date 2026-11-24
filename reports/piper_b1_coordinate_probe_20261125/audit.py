"""Independently recompute reported position scores and compare counterfactual action displacements."""
from pathlib import Path
import hashlib
import json

import numpy as np


def load(p):
    return json.loads(p.read_bytes())


def main(root):
    idx=load(root/'target_index.json')
    labels=load(root/'labels.json')['jaw']
    scores=load(root/'readout_results.json')
    rows=load(root/'heldout_predictions.json')
    prior=root.parent/'piper_b1_vision_contrasts_20261124'
    assert scores['checkpoint_ids']==load(prior/'final.json')['checkpoint_ids']
    label_bytes=(root/'labels.json').read_bytes().replace(b'\r\n',b'\n')
    assert scores['label_sha256']==hashlib.sha256(label_bytes).hexdigest()
    assert len(idx)==84 and sum(r['train'] for r in idx)==50
    prior_metrics=load(prior/'audit.json')['runs']
    groups=0
    test={r['episode'] for r in idx if not r['train']}
    for name,s in scores['scores'].items():
        phase=name.split('/')[1].replace('jaw_','')
        actual={}
        for r in idx:
            key=f"{r['index']:02d}_{phase}"
            if phase=='block':
                actual[r['episode']]=r['wood_uv']
            elif key in labels:
                actual[r['episode']]=np.mean(labels[key],axis=0)
        center=np.mean([x for ep,x in actual.items() if ep not in test],axis=0)
        entries=rows[name]
        assert {e['episode'] for e in entries}==set(actual)&test
        assert len(entries)==len(set(actual)&test)==s['heldout']
        assert len(set(actual)-test)==s['train']
        truth=np.asarray([actual[e['episode']] for e in entries])
        assert np.allclose(truth,[e['target'] for e in entries])
        for method in ['constant','pixels','state_only','layer2_map','layer4_map','pooled','shuffled_pooled']:
            pred=center if method=='constant' else np.asarray([e[method] for e in entries])
            error=np.linalg.norm(pred-truth,axis=1)
            m=s[method]
            assert np.allclose([m['median_px'],m['mean_px'],m['p90_px']],
                               [np.median(error),error.mean(),np.percentile(error,90)],rtol=0,atol=1e-5)
            assert m['within_10px']==int(sum(error<=10))
            assert m['within_15px']==int(sum(error<=15))
            groups+=1
    mask=load(root/'region_results.json')
    for method,m in mask['scores']['intact'].items():
        assert abs(m['mean_px']-scores['scores']['baseline/block'][method]['mean_px'])<1e-6
    traces=load(root/'trace_results.json')
    stats={}
    pairs=load(prior/'final.json')['geometry']['pairs']
    assert len(set(ep for pair in pairs for ep in pair))==32
    decoded={}
    for variant in ('baseline','high','shared'):
        by_ep={e['episode']:e for e in rows[variant+'/block']}
        frame_metrics={}
        for method in ('pixels','state_only','layer2_map','layer4_map','pooled'):
            projections=[]
            gains=[]
            for a,b in pairs:
                actual=np.subtract(by_ep[b]['target'],by_ep[a]['target'])
                pred=np.subtract(by_ep[b][method],by_ep[a][method])
                signed=pred.dot(actual)/np.linalg.norm(actual)
                projections.append(signed)
                gains.append(signed/np.linalg.norm(actual))
            frame_metrics[method]=dict(along_correct_direction=int(sum(x>0 for x in gains)),
                median_projected_px=float(np.median(projections)),median_gain=float(np.median(gains)))
        decoded[variant]=frame_metrics
    for variant,data in traces.items():
        stats[variant]={}
        for view in ('overhead','wrist'):
            directions=[r for r in data['rows'] if r['view']==view]
            assert len(directions)==32
            assert {(r['recipient'],tuple(r['pair'])) for r in directions}=={
                (ep,tuple(pair)) for pair in pairs for ep in pair}
            agg={}
            for k in directions[0]:
                if k not in ('view','recipient','pair'):
                    vals=np.array([r[k] for r in directions])
                    assert np.isfinite(vals).all()
                    agg[k]=dict(median=float(np.median(vals)),maximum=float(vals.max()))
            stats[variant][view]=agg
        actual_max=stats[variant]['overhead']['endpoint_displacement_mm']['maximum']
        assert abs(actual_max-prior_metrics[variant]['direct_target_max_mm'])<2e-5
    return dict(train_episodes=50,heldout_episodes=34,recomputed_score_groups=groups,
                disjoint_pairs=len(pairs),review_frame_pair_decoding=decoded,stage_trace=stats)


if __name__=='__main__':
    out=Path(__file__).resolve().parent
    (out/'audit.json').write_bytes((json.dumps(main(out),indent=1,allow_nan=False)+'\n').encode())
