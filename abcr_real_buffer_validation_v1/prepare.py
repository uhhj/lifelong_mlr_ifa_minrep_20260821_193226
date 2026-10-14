import argparse,collections,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);start=time.time();pilot=a.repo/'reports/abcr_object_pilots_v1';prior=a.repo/'reports/abcr_pair_availability_a_v1'
    write(a.out/'PROTOCOL.json',PROTOCOL,True)
    binding={str(p):sha(p) for p in [pilot/'DATA_MANIFEST.json',pilot/'NORMALIZATION.json',pilot/'FROZEN_STAGE_THRESHOLDS.json',prior/'STAGE_MANIFEST.json',prior/'INPUT_MANIFEST.json']}
    binding.update({p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('common.py')]})
    write(a.out/'PREPARATION_BINDING.json',binding,True)
    manifest=read(pilot/'DATA_MANIFEST.json');action_maps={};demos={}
    for rec in manifest['demos']:
        if rec['task']<6:continue
        raw=pilot/(rec['id']+'_raw.npz');assert sha(raw)==rec['raw_sha256'];z=np.load(raw);act=z['actions'];t=rec['task'];demos[(t,rec['demo'])]=rec
        for w in range(rec['n']):
            key=np.ascontiguousarray(act[np.minimum(np.arange(w,w+8),len(act)-1)],dtype=np.float32).tobytes()
            action_maps.setdefault(t,{}).setdefault(key,[]).append((rec['demo'],w,rec['n']))
    cached=read(prior/'INPUT_MANIFEST.json')['files'];mapping=[]
    for i,r in enumerate(cached):
        p=Path(r['file']);assert p.stat().st_size==r['size'] and sha(p)==r['sha256'],('FEATURE_CHANGED',str(p))
        z=torch.load(p,map_location='cpu');assert tuple(z['feature'].shape)==(1,8,4,512) and tuple(z['actions'].shape)==(1,8,7)
        matches=action_maps[r['task']].get(np.ascontiguousarray(z['actions'].numpy().reshape(8,7),dtype=np.float32).tobytes(),[])
        row=dict(seed=r['seed'],task=r['task'],file=str(p),sha256=r['sha256'],record=p.name,matches=len(matches),demo=None,window_start=None,anchor_t=None,trajectory_length=None)
        if len(matches)==1:
            demo,w,n=matches[0];row.update(demo=demo,window_start=w,anchor_t=min(w+7,n-1),trajectory_length=n)
        mapping.append(row)
        if i%1000==0:print('IDENTITY',i,len(cached),flush=True)
    # The feature saver uses RandomSampler: filenames are export order, never physical time.
    write(a.out/'WINDOW_IDENTITY_MANIFEST.json',dict(records=mapping,unmapped=sum(r['matches']!=1 for r in mapping),method='exact complete action-window content, reject nonunique; source filename is NOT frame time'),True)
    stages=read(prior/'STAGE_MANIFEST.json')['stages'];stages=[s for s in stages if s['task'] in [7,8,9]]
    lookup={(r['seed'],r['task'],r['record']):r for r in mapping}
    for s in stages:
        for name in s['old_files']:
            t,record=name[1:].split('__',1);src=lookup[(s['seed'],int(t),record)];p=Path(s['buffer_dir'])/name
            assert p.exists() and sha(p)==src['sha256'],('BUFFER_CHANGED',str(p))
    write(a.out/'STAGE_MANIFEST.json',dict(stages=stages),True)
    write(a.out/'EXPERIMENT_CONTEXT.json',dict(server_git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo).decode().strip(),server_branch=subprocess.check_output(['git','branch','--show-current'],cwd=a.repo).decode().strip(),source_head='5a85735a2d8214cf5e562dabfd877c177016d402',python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),cpu=platform.processor(),formal_training_resumed=False,seconds=time.time()-start,source_evidence=binding))
    print('PREPARED',len(mapping),'ambiguous',sum(r['matches']!=1 for r in mapping),flush=True)

if __name__=='__main__':main()
