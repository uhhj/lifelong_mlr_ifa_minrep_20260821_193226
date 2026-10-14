"""CPU review supplements: existing-table attribution and raw action identity.

No physical-state classifier, visual recalibration, future-action labels, or
training. Action-window matches are provenance evidence, not state comparability.
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
import h5py
from abcr_pair_availability_a_v1.io import sha, write, table, read_table


def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def literal_assignment(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def contributions(base,out):
    rows=read_table(base/'by_old_task.csv')
    totals=collections.defaultdict(lambda:collections.Counter())
    for r in rows:
        key=(r['seed'],r['current_task'],r['visual_threshold'])
        for f in ('matched','share','bif'):totals[key][f]+=int(r[f])
    result=[]
    for r in rows:
        t=totals[(r['seed'],r['current_task'],r['visual_threshold'])]
        item=dict(r)
        for f in ('matched','share','bif'):
            item[f+'_contribution']=int(r[f])/t[f] if t[f] else None
        result.append(item)
    table(out/'OLD_TASK_CONTRIBUTIONS.csv',result)
    keyrows=[r for r in result if r['current_task']=='8' and r['old_task']=='6' and float(r['visual_threshold'])==.7]
    write(out/'TASK8_OLD6_CONCENTRATION.json',dict(rows=keyrows,
        note='Current task8 versus historical task6 only; no reverse/future replay experiment inferred'))


def mapping(root,base,out):
    names=literal_assignment(root/'libero/libero/benchmark/libero_suite_task_map.py','libero_task_map')['libero_object']
    orders=literal_assignment(root/'libero/libero/benchmark/__init__.py','task_orders')
    assert orders[0]==list(range(10))
    manifests=json.loads((base/'INPUT_MANIFEST.json').read_text())
    result=[];asset_rows=[];vectors={}
    for task in (6,7,8,9):
        p=root/'data/libero_object'/(names[task]+'_demo.hdf5')
        index=collections.defaultdict(list)
        with h5py.File(p,'r') as f:
            demos=sorted(f['data'],key=lambda s:int(s.split('_')[-1]))
            keys=set()
            for demo in demos:
                d=f['data'][demo];a=d['actions'][:].astype(np.float32)
                assert a.ndim==2 and a.shape[1]==7
                keys.update(d['obs'].keys())
                asset_rows.append(dict(task=task,file=str(p),file_size=p.stat().st_size,demo=demo,length=len(a),
                    float32_actions_sha256=arrsha(a),obs_shapes={k:list(v.shape) for k,v in d['obs'].items()}))
                for start in range(len(a)):
                    idx=np.minimum(np.arange(start,start+8),len(a)-1)
                    seq=np.ascontiguousarray(a[idx])
                    index[seq.tobytes()].append((demo,start,min(start+7,len(a)-1),max(0,start+8-len(a))))
            for entry in manifests['files']:
                if entry['task']!=task:continue
                path=Path(entry['file'])
                assert sha(path)==entry['sha256'],'INPUT_CHANGED_SINCE_A'
                payload=torch.load(str(path),map_location='cpu',weights_only=True)
                a=payload['actions'].numpy()[0]
                h=payload['feature'][0,-1]
                desc=torch.cat([torch.nn.functional.normalize(h[i],dim=0,eps=1e-8) for i in (1,2)])/math.sqrt(2)
                vectors[(entry['seed'],task,entry['source_record_index'])]=desc
                assert a.shape==(8,7) and a.dtype==np.float32
                hits=index.get(np.ascontiguousarray(a).tobytes(),[])
                unique=len(hits)==1
                result.append(dict(seed=entry['seed'],task=task,record_id=entry['source_record_index'],feature_file=str(path),
                    feature_sha256=entry['sha256'],action_window_sha256=arrsha(a),hdf5_file=str(p),
                    match_count=len(hits),match_status='UNIQUE_ACTION_WINDOW' if unique else 'AMBIGUOUS' if hits else 'UNMATCHED',
                    trajectory_id=hits[0][0] if unique else None,window_start=hits[0][1] if unique else None,
                    last_step_anchor=hits[0][2] if unique else None,padded_tail_steps=hits[0][3] if unique else None,
                    all_matching_locations=json.dumps(hits,separators=(',',':'))))
    table(out/'ACTION_WINDOW_PROVENANCE.csv',result)
    write(out/'RAW_HDF5_ASSET_MANIFEST.json',dict(assets=asset_rows,
        scope='Actions of all50 demos for task6/7/8/9; raw images not decoded or feature re-exported',
        dataset_source=str(root/'libero/lifelong/datasets.py'),dataset_source_sha256=sha(root/'libero/lifelong/datasets.py'),
        feature_export_source_sha256=sha(root/'libero/lifelong/models/base_policy.py'),
        matching_rule='Exact bytes after native float64->float32 cast, seq_length8, frame_stack1, repeated last-step padding',
        uniqueness_scope='all50 trajectories of the corresponding task',
        limitation='Unique action-window content identifies a source candidate; latent pixels/augmentation not independently matched. This is not physical-state validation.'))
    counts=collections.Counter(r['match_status'] for r in result)
    # Attach identities only to the already-frozen stratified audit examples.
    lookup={(r['seed'],r['task'],r['record_id']):r for r in result}
    pairs=[]
    for seed in (42,43,44):
        for task in (7,8,9):
            for r in read_table(base/('object_seed%d_task%d'%(seed,task))/'audit_sample.csv'):
                ni=int(Path(r['new_file']).stem);oi=int(Path(r['old_file']).stem.split('__')[1]);old=int(r['old_task'])
                a=lookup[(seed,task,ni)];b=lookup[(seed,old,oi)]
                pairs.append(dict(seed=seed,current_task=task,old_task=old,visual_threshold=r['visual_threshold'],label_A=r['class'],
                    new_record=ni,old_record=oi,new_match_status=a['match_status'],old_match_status=b['match_status'],
                    new_trajectory=a['trajectory_id'],old_trajectory=b['trajectory_id'],
                    new_anchor=a['last_step_anchor'],old_anchor=b['last_step_anchor'],
                    priority_task8_old6=task==8 and old==6,
                    physical_match_verified=False))
    table(out/'B_PREPARATION_AUDIT_PAIRS.csv',pairs)
    summary=dict(status='IDENTITY_PREPARATION_ONLY',counts=dict(counts),total=len(result),
        priority='task8 current / task6 historical',paired_audit_rows=len(pairs),
        audit_pairs_both_unique=sum(r['new_match_status']==r['old_match_status']=='UNIQUE_ACTION_WINDOW' for r in pairs),
        physical_labels='NOT_RUN',unmodulated_visual_features='NOT_EXPORTED',human_audit='NOT_RUN',
        missing=['frozen state-distance definition and tau_s','frozen unmodulated visual threshold','independent calibration split','human annotation setup'],
        note='Do not turn record filenames into trajectory IDs; ambiguous/no-match records remain unresolved.',
        formal_training_resumed=False,cuda_initialized=torch.cuda.is_initialized())
    write(out/'B_PREPARATION.json',summary)
    print('B_PREPARATION',json.dumps(summary,ensure_ascii=False),flush=True)
    # Descriptive distributions only, registered here before computing any values.
    quantiles=[0,.01,.05,.25,.5,.75,.9,.95,.99,.999,1]
    write(out/'DISTRIBUTION_PROTOCOL.json',dict(quantiles=quantiles,metric='unchanged A last-step mean two-view signed cosine',
        within='all unordered distinct records within each task; overlapping windows remain included',
        cross='all current-record x actual A stage old-buffer members',threshold_selection=False,
        warning='No physical-state labels, no causal attribution to FiLM versus scene/object effects'))
    stages=json.loads((base/'STAGE_MANIFEST.json').read_text())['stages'];stats=[]
    def add(seed,current,old,kind,values):
        values=np.asarray(values,dtype=np.float32).reshape(-1)
        assert np.isfinite(values).all()
        row=dict(seed=seed,current_task=current,old_task=old,kind=kind,pairs=len(values),mean=float(values.mean()),std=float(values.std()))
        for q,v in zip(quantiles,np.quantile(values,quantiles)):row['q_'+str(q)]=float(v)
        stats.append(row)
    for seed in (42,43,44):
        for task in (6,7,8,9):
            records=[r for r in manifests['files'] if r['seed']==seed and r['task']==task]
            x=torch.stack([vectors[(seed,task,r['source_record_index'])] for r in records])
            v=(x@x.T).numpy();ii,jj=np.triu_indices(len(x),1)
            add(seed,task,task,'WITHIN_TASK_DISTINCT_RECORDS',v[ii,jj])
            if task==6:continue
            stage=next(s for s in stages if s['seed']==seed and s['task']==task)
            ids=[(int(name.split('__')[0][1:]),int(Path(name).stem.split('__')[1])) for name in stage['old_files']]
            y=torch.stack([vectors[(seed,t,i)] for t,i in ids]);v=(x@y.T).numpy()
            add(seed,task,'all','CROSS_TASK_STAGE_BUFFER',v)
            for old in range(6,task):
                mask=np.array([t==old for t,i in ids]);add(seed,task,old,'CROSS_TASK_BY_OLD',v[:,mask])
    table(out/'VISUAL_SIMILARITY_QUANTILES.csv',stats)
    print('SIMILARITY_DISTRIBUTIONS_COMPLETE',len(stats),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();base=root/'reports/abcr_pair_availability_a_v1';out=root/'reports/abcr_pair_availability_a_v1_review_v1';out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    contributions(base,out);mapping(root,base,out)


if __name__=='__main__':main()
