"""Role-isolated native frontend extraction for the frozen six-task panel.

No policy optimization; the same native A_train augmentation is retained.
"""
import argparse
import time
import numpy as np
from .common import *

CONF_OUT=OUT/'confirmation'
CONF_STORE=STORE/'confirmation'

def main(seed,task,role):
    import h5py
    import torch
    from cb_ifa_experiment import gpu_runtime as env
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    from libero.lifelong.models.complete_policy import CompletePolicy
    from libero.lifelong.utils import get_task_embs
    from torch.utils.data._utils.collate import default_collate
    from robomimic.utils.tensor_utils import map_tensor
    assert seed in read(OUT/'TASK_PANEL_MANIFEST.json')['seeds']
    entry=read(OUT/'CONFIRMATION_DATA_MANIFEST.json')['tasks'][str(task)]
    template=ROOT/'runs/fixed_kappa_confirmation_v2/feature_seed45/run_001/config.json'
    cfg=env.remap_config(read(template));cfg.seed=seed;cfg.benchmark_name=entry['suite'];cfg.use_wandb=False
    cfg.experiment_dir=str(CONF_STORE/'cache_bootstrap');cfg.feature_dir=str(CONF_STORE/'features')
    assert cfg.aug and cfg.policy.policy_type=='CompletePolicy'
    torch.set_num_threads(2);env.seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=CONF_STORE/'views'/('%s_%s.hdf5'%(task,role));view.parent.mkdir(parents=True,exist_ok=True)
    if not view.exists():
        temp=view.with_name(view.name+'.tmp')
        with h5py.File(temp,'w') as f:
            data=f.create_group('data')
            for demo in entry['demos'][role]:data[demo]=h5py.ExternalLink(entry['path'],'/data/'+demo)
        os.replace(temp,view)
    with h5py.File(view,'r') as f:assert set(f['data'])==set(entry['demos'][role])
    raw,shape=get_dataset(str(view),dict(cfg.data.obs.modality),seq_len=8,frame_stack=1,demo_limit=None,hdf5_cache_mode='low_dim')
    tokens=get_task_embs(cfg,[entry['language']]);ds=SequenceVLDataset(raw,tokens['input_ids'][0],tokens['attention_mask'][0]);cfg.shape_meta=shape
    model=CompletePolicy(cfg,shape).cuda();model.train(role=='A_train');model.img_aug.train(role=='A_train')
    checkpoint=torch.load(cfg.pretrain,map_location='cpu')['state_dict'];checkpoint={k[7:] if k.startswith('module.') else k:v for k,v in checkpoint.items()}
    assert all(k in checkpoint and torch.equal(v.cpu(),checkpoint[k]) for k,v in model.state_dict().items())
    fingerprint=env.digest({n:v for n,v in model.state_dict().items() if not n.startswith(('temporal_transformer.','policy_head.'))});del checkpoint
    target=CONF_STORE/'features'/('seed%d'%seed)/('task%d'%task)/role;target.mkdir(parents=True,exist_ok=True)
    rows=entry['windows'][role];begin=time.monotonic()
    for ci in range((len(rows)+63)//64):
        path=target/('chunk%04d.npz'%ci);metadata=path.with_suffix('.json');selected=rows[ci*64:(ci+1)*64]
        if path.exists() and metadata.exists():
            m=read(metadata);assert m['rows']==selected and m['file_sha256']==sha(path) and m['encoder_sha256']==fingerprint;continue
        ff=[];aa=[]
        for row in selected:
            index=raw._demo_id_to_start_indices[row['trajectory_id']]+row['window_index'];assert raw._index_to_demo_id[index]==row['trajectory_id']
            env.seed_all(stable_seed('task_relation_signal_screen_v2','cache',seed,entry['canonical_id'],role,row['source_record_index']))
            data=map_tensor(default_collate([ds[index]]),lambda x:x.cuda())
            with torch.no_grad():prepared=model.preprocess_input(data);feature=model(prepared)
            assert tuple(feature.shape)==(1,8,4,512) and tuple(prepared['actions'].shape)==(1,8,7)
            assert torch.isfinite(feature).all() and torch.isfinite(prepared['actions']).all()
            ff.append(feature.cpu().numpy()[0]);aa.append(prepared['actions'].cpu().numpy()[0])
        npz(path,feature=np.stack(ff),actions=np.stack(aa))
        write(metadata,dict(seed=seed,task=task,role=role,rows=selected,encoder_sha256=fingerprint,file_sha256=sha(path),template_path=str(template),template_sha256=sha(template),task_language=entry['language'],augmentation=role=='A_train',lossless_float32=True,optimizer_steps=0))
        write(CONF_OUT/'CACHE_STATUS.json',dict(seed=seed,task=task,role=role,completed_windows=min((ci+1)*64,len(rows)),total_windows=len(rows),seconds=time.monotonic()-begin,formal_training_resumed=False))
        print('CACHE',seed,task,role,ci,len(selected),round(time.monotonic()-begin,2),flush=True)
    write(CONF_OUT/('CACHE_s%d_t%d_%s.json'%(seed,task,role)),dict(status='PASS',encoder_sha256=fingerprint,windows=len(rows),seconds=time.monotonic()-begin,optimizer_steps=0,formal_training_resumed=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--role',choices=['A_train','D_signal','E_eval'],required=True)
    args=parser.parse_args();main(args.seed,args.task,args.role)
