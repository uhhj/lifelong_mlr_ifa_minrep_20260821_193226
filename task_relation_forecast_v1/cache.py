"""Native frozen-frontend windows, with trajectory roles kept separate.

Only feature preparation. No optimizer is constructed and no training update
is performed. Each completed chunk can be resumed independently.
"""
import argparse
import os
import shutil
import time
import numpy as np
from .common import ROOT, OUT, STORE, read, write, sha, stable_seed


def main(seed, task, smoke=False):
    from cb_ifa_experiment import gpu_runtime as env
    import torch
    from torch.utils.data._utils.collate import default_collate
    from libero.lifelong.datasets import get_dataset, SequenceVLDataset
    from libero.lifelong.models.complete_policy import CompletePolicy
    from libero.lifelong.utils import get_task_embs
    config_source = ROOT/('runs/fixed_kappa_confirmation_v2/feature_seed%d/run_001/config.json'%seed)
    cfg = env.remap_config(read(config_source))
    cfg.seed = seed
    cfg.experiment_dir = str(STORE/'cache_bootstrap')
    cfg.feature_dir = str(STORE/'features')
    cfg.use_wandb = False
    assert cfg.aug and cfg.policy.policy_type == 'CompletePolicy'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env.seed_all(seed)
    entry = read(OUT/'DATA_MANIFEST.json')['tasks'][str(task)]
    raw, shape = get_dataset(entry['path'], dict(cfg.data.obs.modality), seq_len=8,
                             frame_stack=1, demo_limit=None, hdf5_cache_mode='low_dim')
    assert sum(entry['lengths'].values()) == len(raw)
    tokens = get_task_embs(cfg, [entry['language']])
    ds = SequenceVLDataset(raw, tokens['input_ids'][0], tokens['attention_mask'][0])
    cfg.shape_meta = shape
    model = CompletePolicy(cfg, shape).cuda()
    # Native preload must supply every frozen frontend tensor.
    ck = torch.load(cfg.pretrain, map_location='cpu')['state_dict']
    ck = {k[7:] if k.startswith('module.') else k: v for k, v in ck.items()}
    for name, value in model.state_dict().items():
        assert name in ck and torch.equal(value.cpu(), ck[name]), ('Checkpoint bootstrap mismatch', name)
    del ck
    fingerprint = env.digest({n: v for n, v in model.state_dict().items()
                              if not n.startswith(('temporal_transformer.', 'policy_head.'))})
    start = time.monotonic()
    for role in ('A_train', 'D_signal', 'E_eval'):
        rows = entry['windows'][role]
        model.train(role == 'A_train')
        # Same native image augmentation module; frozen CLIP/FiLM weights and
        # decoder are not optimized. Only training windows enable augmentation.
        model.img_aug.train(role == 'A_train')
        role_dir = STORE/'features'/('seed%d'%seed)/('task%d'%task)/role
        role_dir.mkdir(parents=True, exist_ok=True)
        chunks = range(1) if smoke else range((len(rows)+63)//64)
        for ci in chunks:
            path = role_dir/('smoke.npz' if smoke else 'chunk%04d.npz'%ci)
            meta_path = path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                meta = read(meta_path)
                assert meta['encoder_sha256'] == fingerprint and meta['file_sha256'] == sha(path)
                continue
            if shutil.disk_usage(STORE).free < 2*1024**3:
                raise RuntimeError('STORAGE_RESERVE_REACHED: require 2GiB free; no historical cleanup from runner')
            selected = rows[:1] if smoke else rows[ci*64:(ci+1)*64]
            ff, aa = [], []
            for row in selected:
                idx = row['source_record_index']
                # This must agree with the native dataset's actual demo mapping.
                assert raw._index_to_demo_id[idx] == row['trajectory_id']
                assert idx - raw._demo_id_to_start_indices[row['trajectory_id']] == row['window_index']
                data = default_collate([ds[idx]])
                env.seed_all(stable_seed('task_relation_forecast_v1', 'cache', seed, task, role, idx))
                data = {k: v for k, v in data.items()}
                from robomimic.utils.tensor_utils import map_tensor
                data = map_tensor(data, lambda t: t.cuda())
                with torch.no_grad():
                    prepared = model.preprocess_input(data)
                    feature = model(prepared)
                assert tuple(feature.shape) == (1, 8, 4, 512)
                assert tuple(prepared['actions'].shape) == (1, 8, 7)
                assert torch.isfinite(feature).all() and torch.isfinite(prepared['actions']).all()
                ff.append(feature.cpu().numpy()[0]); aa.append(prepared['actions'].cpu().numpy()[0])
            tmp = path.with_name(path.name+'.tmp.npz')
            np.savez_compressed(tmp, feature=np.stack(ff), actions=np.stack(aa))
            os.replace(tmp, path)
            with np.load(path) as check:
                assert np.array_equal(check['feature'], np.stack(ff))
            write(meta_path, dict(seed=seed, task=task, role=role, rows=selected,
                                  encoder_sha256=fingerprint, file_sha256=sha(path),
                                  native_config_source=str(config_source), native_config_sha256=sha(config_source),
                                  task_language=entry['language'], augmentation=role=='A_train',
                                  lossless_float32=True, optimizer_steps=0))
            write(OUT/'CACHE_STATUS.json', dict(stage='FEATURE_PREPARATION', seed=seed,task=task,role=role,
                                               chunk=ci,windows_done=min((ci+1)*64,len(rows)),
                                               smoke=smoke,elapsed_seconds=time.monotonic()-start,
                                               diagnostic_training_performed=False,formal_training_resumed=False))
            print('CACHE',seed,task,role,ci,len(selected),round(time.monotonic()-start,2),flush=True)
    write(OUT/('CACHE_%s_s%d_t%d.json'%('SMOKE' if smoke else 'COMPLETE',seed,task)),
          dict(status='PASS', encoder_sha256=fingerprint, seconds=time.monotonic()-start,
               optimizer_steps=0, formal_training_resumed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    main(args.seed,args.task,args.smoke)
