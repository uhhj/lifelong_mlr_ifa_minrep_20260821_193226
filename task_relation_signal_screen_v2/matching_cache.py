"""Base visual embeddings BEFORE language FiLM, D_signal windows only."""
import time
import numpy as np
from .common import *

def main(task):
    import torch
    from torch.utils.data._utils.collate import default_collate
    from cb_ifa_experiment import gpu_runtime as env
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    from libero.lifelong.models.complete_policy import CompletePolicy
    from libero.lifelong.utils import get_task_embs
    from robomimic.utils.tensor_utils import map_tensor
    dest=STORE/'matching'/('task%d.npz'%task);meta=OUT/'matching'/('task%d.json'%task)
    if meta.exists():assert sha(dest)==read(meta)['sha256'];return
    begin=time.monotonic();entry=read(OLD/'DATA_MANIFEST.json')['tasks'][str(task)]
    cfg=env.remap_config(read(ROOT/'runs/fixed_kappa_confirmation_v2/feature_seed45/run_001/config.json'))
    cfg.seed=45;cfg.use_wandb=False;cfg.experiment_dir=str(STORE/'matching_bootstrap');cfg.feature_dir=str(STORE/'matching')
    torch.set_num_threads(2);env.seed_all(45)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Restrict even the dataset constructor/cache to D_signal, via read-only external links.
    import h5py
    view=STORE/'matching'/('task%d_D_only.hdf5'%task);view.parent.mkdir(parents=True,exist_ok=True)
    if not view.exists():
        with h5py.File(view,'w') as dst:
            group=dst.create_group('data')
            for demo in entry['demos']['D_signal']:group[demo]=h5py.ExternalLink(entry['path'],'/data/'+demo)
    with h5py.File(view,'r') as check:assert set(check['data'])==set(entry['demos']['D_signal'])
    raw,shape=get_dataset(str(view),dict(cfg.data.obs.modality),seq_len=8,frame_stack=1,demo_limit=None,hdf5_cache_mode=None)
    tokens=get_task_embs(cfg,[entry['language']]);ds=SequenceVLDataset(raw,tokens['input_ids'][0],tokens['attention_mask'][0]);cfg.shape_meta=shape
    model=CompletePolicy(cfg,shape).cuda().eval();model.img_aug.eval()
    ck=torch.load(cfg.pretrain,map_location='cpu')['state_dict'];ck={k[7:] if k.startswith('module.') else k:v for k,v in ck.items()}
    assert all(n in ck and torch.equal(v.cpu(),ck[n]) for n,v in model.state_dict().items());del ck
    values=[]
    for row in entry['windows']['D_signal']:
        idx=raw._demo_id_to_start_indices[row['trajectory_id']]+row['window_index'];assert raw._index_to_demo_id[idx]==row['trajectory_id']
        data=map_tensor(default_collate([ds[idx]]),lambda t:t.cuda())
        with torch.no_grad():
            prepared=model.preprocess_input(data)
            images=prepared['obs']['agentview_rgb'];images=images.reshape(-1,*images.shape[2:])
            visual=model.image_proj(model.image_encoders(images,langs=None))
            values.append(visual[:row['valid_frame_count']].double().mean(0).cpu().numpy())
    npz(dest,visual=np.stack(values))
    write(meta,dict(task=task,rows=entry['windows']['D_signal'],source_checkpoint=cfg.pretrain,source_checkpoint_sha256=sha(cfg.pretrain),representation='native agent-view image encoder+image_proj, before FiLM; valid-frame mean; no language in matching distance',sha256=sha(dest),seconds=time.monotonic()-begin,role='D_signal',hdf5_view=str(view),allowed_demos=entry['demos']['D_signal'],E_eval_accessed=False,formal_training_resumed=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);main(p.parse_args().task)
