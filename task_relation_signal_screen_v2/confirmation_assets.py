"""Fetch only the six text-selected official datasets; freeze identity and roles.

This command never constructs an optimizer, trains, or reads rollout outcomes.
"""
import hashlib
import os
import shutil
import time
import numpy as np
from .common import *

REPO='yifengzhu-hf/LIBERO-datasets'

def download(item,revision,path):
    import requests
    expected=item['lfs']['oid'];size=item['size']
    if path.exists():assert path.stat().st_size==size and sha(path)==expected;return
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.name+'.partial')
    url='https://huggingface.co/datasets/%s/resolve/%s/%s'%(REPO,revision,item['path'])
    for attempt in range(3):
        offset=partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url,headers={'Range':'bytes=%d-'%offset} if offset else {},stream=True,timeout=(20,60)) as response:
                response.raise_for_status()
                if offset and response.status_code==206:
                    assert response.headers['Content-Range'].startswith('bytes %d-'%offset)
                    mode='ab'
                else:mode='wb'
                with partial.open(mode) as f:
                    for chunk in response.iter_content(8*1024*1024):
                        if chunk:f.write(chunk)
                    f.flush();os.fsync(f.fileno())
            assert partial.stat().st_size==size and sha(partial)==expected, 'OFFICIAL_LFS_CONTENT_MISMATCH'
            os.replace(partial,path);return
        except Exception:
            if attempt==2:raise
            time.sleep(2)

def main():
    import requests
    import h5py
    from task_relation_forecast_v1.common import balanced_windows
    proposal=read(OUT/'TASK_PANEL_PROPOSAL.json');registry=OUT/'DATASET_DOWNLOAD_REGISTRY.json'
    if not registry.exists():
        response=requests.get('https://huggingface.co/api/datasets/'+REPO,timeout=30);response.raise_for_status();revision=response.json()['sha']
        response=requests.get('https://huggingface.co/api/datasets/%s/tree/%s?recursive=true&expand=false'%(REPO,revision),timeout=30);response.raise_for_status()
        mapping={r['path']:r for r in response.json()}
        items=[mapping[r['dataset_relative_path']] for r in proposal['selected']]
        freeze(registry,dict(repository=REPO,revision=revision,files=items,selection_sha256=sha(OUT/'TASK_PANEL_PROPOSAL.json'),source_reference='repository libero/libero/utils/download_utils.py HF_REPO_ID',formal_training_resumed=False))
    acquisition=read(registry);items=acquisition['files']
    required=sum(i['size'] for i in items if not (STORE/'confirmation/datasets'/i['path']).exists())
    assert shutil.disk_usage(STORE).free>required+15*1024**3, 'CONFIRMATION_STORAGE_RESERVE'
    for item in items:
        write(OUT/'CONFIRMATION_ASSET_STATUS.json',dict(stage='DOWNLOAD',file=item['path'],formal_training_resumed=False))
        download(item,acquisition['revision'],STORE/'confirmation/datasets'/item['path'])
    # Targeted comparison to all ten OBJECT datasets, including the six base tasks.
    from cb_ifa_experiment import gpu_runtime as env
    from libero.libero.benchmark import get_benchmark
    # Historical configuration files exist for45/46/47; this is only a path/interface
    # audit, not initialization of a new optimizer or an invented seed48 config.
    cfg=env.train_config(45,STORE/'confirmation/configuration_only');bench=get_benchmark('libero_object')(0)
    historical={}
    for i in range(bench.n_tasks):
        path=ROOT/'data'/bench.get_task_demonstration(i)
        with h5py.File(path,'r') as f:
            for demo in f['data']:
                h=hashlib.sha256(f['data'][demo]['actions'][:].tobytes()).hexdigest()
                historical[h]=dict(task=i,demo=demo,path=str(path))
    tasks={};duplicate=[]
    for number,entry in enumerate(proposal['selected']):
        key=entry['canonical_id'];path=STORE/'confirmation/datasets'/entry['dataset_relative_path']
        with h5py.File(path,'r') as f:
            all_demos=sorted(f['data'],key=lambda d:int(d.split('_')[-1]));offset=0;offsets={};lengths={};fingerprints={}
            for demo in all_demos:
                actions=f['data'][demo]['actions'];assert actions.ndim==2 and actions.shape[1]==7 and len(actions)>0
                offsets[demo]=offset;lengths[demo]=len(actions);offset+=len(actions)
                fingerprints[demo]=hashlib.sha256(actions[:].tobytes()).hexdigest()
                if fingerprints[demo] in historical:duplicate.append(dict(task=key,demo=demo,original=historical[fingerprints[demo]]))
                for field in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states','ee_pos','ee_ori'):
                    assert field in f['data'][demo]['obs'], ('OBSERVATION_MISSING',key,field)
            ordered=sorted(all_demos,key=lambda d:(stable_seed('screen_v2',key,'split',d),d))[:50]
            n=len(ordered);assert n>=5
            a,d=(30,10) if n==50 else (int(.6*n),int(.2*n))
            roles=dict(A_train=ordered[:a],D_signal=ordered[a:a+d],E_eval=ordered[a+d:]);windows={}
            for role,demos in roles.items():
                selected=[(demo,i) for demo in demos for i in range(lengths[demo])] if role=='A_train' else balanced_windows(demos,lengths,128 if role=='D_signal' else 320)
                windows[role]=[dict(task=number,canonical_id=key,trajectory_id=demo,window_index=i,source_record_index=offsets[demo]+i,trajectory_length=lengths[demo],valid_frame_count=min(8,lengths[demo]-i)) for demo,i in selected]
        tasks[str(number)]=dict(entry,path=str(path),file_sha256=sha(path),demos=roles,lengths=lengths,action_sha256=fingerprints,windows=windows)
    write(OUT/'CONFIRMATION_DUPLICATE_AUDIT.json',dict(exact_action_duplicates=duplicate,comparison='all selected demos against ten OBJECT tasks; normalized BDDL checked in proposal; same objects or primitive actions do not alone imply same task',formal_training_resumed=False))
    assert not duplicate, 'EXACT_DEMONSTRATION_DUPLICATE'
    # Only materialized previous model directories indicate prior optimization use.
    seed_evidence={}
    for seed in (48,49,50):
        matches=[]
        for root in (ROOT/'diagnostic_storage',ROOT/'runs'):
            for experiment in root.iterdir():
                if not experiment.is_dir() or experiment==STORE:continue
                for pattern in ('seed%d*'%seed,'s%d_*'%seed,'sources/s%d_*'%seed,'transfers/s%d_*'%seed):
                    matches.extend(str(p) for p in experiment.glob(pattern))
        seed_evidence[str(seed)]=sorted(set(matches))
    write(OUT/'CONFIRMATION_SEED_AUDIT.json',dict(seeds=seed_evidence,scope='bounded top-level experiment/seed and sources/transfers directories; no global file scan',formal_training_resumed=False))
    assert not any(seed_evidence.values()), 'REQUESTED_CONFIRMATION_SEED_ALREADY_MATERIALIZED_REVIEW_REQUIRED'
    freeze(OUT/'CONFIRMATION_DATA_MANIFEST.json',dict(tasks=tasks,selection_sha256=sha(OUT/'TASK_PANEL_PROPOSAL.json'),download_registry_sha256=sha(registry),role_rule='whole-demo SHA256 split, 30/10/10 for 50, 60/20/20 otherwise; all training and signal/evaluation roles separate',formal_training_resumed=False))
    freeze(OUT/'TASK_PANEL_MANIFEST.json',dict(status='FROZEN',tasks=[dict(task=i,**e) for i,e in enumerate(proposal['selected'])],seeds=[48,49,50],data_sha256=sha(OUT/'CONFIRMATION_DATA_MANIFEST.json'),pair_count=30,seed_pair_count=90,source_models=18,rollouts=9360,shared_official_base=True,base_original_job_log_available=False,formal_training_resumed=False))
    write(OUT/'CONFIRMATION_ASSET_STATUS.json',dict(status='ASSETS_FROZEN_TRAINING_NOT_STARTED',formal_training_resumed=False))

if __name__=='__main__':main()
