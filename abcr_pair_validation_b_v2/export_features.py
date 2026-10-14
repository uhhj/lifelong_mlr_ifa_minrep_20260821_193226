"""Export native checkpoint pre-FiLM image_proj output; no policy / CUDA / training.

Resume requires identical input, checkpoint, native source and exporter hashes.
Base records are direct HDF5 anchors, not claims about incremental .pt alignment.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import logging as hf_logging

from abcr_pair_validation_b_v1.original_experiment_b import dump_json, sha256


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.npz')
    with open(temp, 'wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == set(arrays)


def raw_tensor(rgb):
    # robomimic ImageModality._default_obs_processor / process_frame:
    # raw OpenGL HWC -> float / 255 -> clip -> CHW; no display-only flip.
    return torch.from_numpy(np.stack(rgb)).float().div(255).clamp(0, 1).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.repo, args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'anchor_features'
    cache.mkdir(exist_ok=True)
    manifest = root / 'reports/abcr_pair_validation_b_v1/BASE_RECORD_MANIFEST.json'
    assets = root / 'reports/abcr_pair_validation_b_v1/BASE_ASSET_MANIFEST.json'
    checkpoint = root / 'checkpoints/object_multitask_model_ep99.pth'
    rgb_source = root / 'libero/lifelong/models/modules/rgb_modules.py'
    policy_source = root / 'libero/lifelong/models/complete_policy.py'
    records = json.loads(manifest.read_text())['records']
    binding = dict(manifest_sha256=sha256(manifest), assets_sha256=sha256(assets),
                   checkpoint_sha256=sha256(checkpoint), rgb_source_sha256=sha256(rgb_source),
                   policy_source_sha256=sha256(policy_source), exporter_sha256=sha256(__file__))
    assert binding['checkpoint_sha256'] == '159351b1df166dd5391c6b1e1317c0ab299fc8b1447fe466acef14ed7896bb7b'
    if (out / 'FEATURE_BINDING.json').exists():
        assert json.loads((out / 'FEATURE_BINDING.json').read_text()) == binding, 'CACHE_BINDING_MISMATCH'
    else:
        dump_json(out / 'FEATURE_BINDING.json', binding)
    checked = []
    for asset in json.loads(assets.read_text())['assets']:
        p = Path(asset['path'])
        assert p.stat().st_size == asset['bytes']
        assert sha256(p) == asset['sha256'], str(p)
        checked.append(dict(path=str(p), sha256=asset['sha256']))
    dump_json(out / 'INPUT_ASSET_CHECK.json', dict(status='PASS', assets=checked), overwrite=True)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    hf_logging.set_verbosity_error()
    started = time.perf_counter()
    cp = torch.load(checkpoint, map_location='cpu')
    cfg = cp['cfg']
    rank, alpha, embed = cfg.policy.rank, cfg.policy.alpha, cfg.policy.embed_size
    state = cp['state_dict']
    enc_state = {k[len('module.image_encoders.'):]: v for k, v in state.items()
                 if k.startswith('module.image_encoders.')}
    proj_state = {k[len('module.image_proj.'):]: v for k, v in state.items()
                  if k.startswith('module.image_proj.')}
    del cp, state, cfg
    gc.collect()
    spec = importlib.util.spec_from_file_location('abcr_native_rgb', rgb_source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    encoder = get_peft_model(native.CLIPVisionEncoder(vision_model_name='openai/clip-vit-base-patch16'),
                            LoraConfig(r=rank, lora_alpha=alpha, target_modules=['q_proj', 'v_proj']))
    projection = torch.nn.Linear(768, embed)
    encoder.load_state_dict(enc_state, strict=True)
    projection.load_state_dict(proj_state, strict=True)
    encoder.eval()
    projection.eval()
    encoder.requires_grad_(False)
    projection.requires_grad_(False)
    n_keys = len(enc_state) + len(proj_state)
    del enc_state, proj_state
    gc.collect()
    bootstrap_seconds = time.perf_counter() - started
    images_checked = 0
    features = []
    new_count = 0
    for index, rec in enumerate(records):
        entry = cache / ('%04d.npz' % index)
        with h5py.File(rec['hdf5_file'], 'r') as h:
            demo = h['data'][rec['demo']]
            t = rec['anchor_t']
            rgb = [demo['obs'][key][t] for key in ('agentview_rgb', 'eye_in_hand_rgb')]
            for key, pixels in zip(('agentview_rgb', 'eye_in_hand_rgb'), rgb):
                assert hashlib.sha256(pixels.tobytes()).hexdigest() == rec['rgb_sha256'][key]
                images_checked += 1
            assert np.array_equal(demo['actions'][t:t+3].astype(np.float32), np.array(rec['future_actions'], np.float32))
            robot = demo['robot_states'][t]
            assert np.array_equal(robot[2:5], np.array(rec['eef_pos_m']))
            assert np.array_equal(robot[5:9], np.array(rec['eef_quat_xyzw']))
            assert float((robot[0]-robot[1])/.08) == rec['gripper_open_fraction']
        if entry.exists():
            with np.load(entry, allow_pickle=False) as z:
                assert str(z['record_id']) == rec['id']
                feat = z['features']
        else:
            image = raw_tensor(rgb)
            with torch.inference_mode():
                feat_t = projection(encoder(image, langs=None))
                if index == 0:
                    from robomimic.utils.obs_utils import ImageModality
                    official_input = ImageModality._default_obs_processor(torch.from_numpy(np.stack(rgb)))
                    assert torch.equal(image, official_input)
                    repeat = projection(encoder(image, langs=None))
                    bare = encoder.base_model.model
                    direct = projection(bare.clip(pixel_values=bare.clip_transforms(image)).pooler_output)
                    assert torch.equal(feat_t, repeat)
                    assert torch.equal(feat_t, direct)
                    dump_json(out / 'FEATURE_PARITY.json', dict(status='PASS', checkpoint_keys_strict=n_keys,
                        rank=rank, lora_alpha=alpha, output_dim=embed, native_input_bitwise_equal=True,
                        native_forward_bitwise_equal=True, repeated_bitwise_equal=True,
                        dropout='eval', rgb_flip=False, cpu_only=True, torch=str(torch.__version__),
                        feature_location='image_proj(image_encoders(raw_image)), BEFORE FiLM',
                        bootstrap_seconds=bootstrap_seconds), overwrite=True)
            feat = feat_t.numpy().copy()
            assert feat.shape == (2, 512) and np.isfinite(feat).all()
            assert (np.linalg.norm(feat, axis=1) > 0).all()
            atomic_npz(entry, features=feat, record_id=np.array(rec['id']))
            new_count += 1
        features.append(feat)
        if (index+1) % 10 == 0 or index+1 == len(records):
            status = dict(status='EXPORTING', completed=index+1, total=len(records), new_count=new_count,
                          seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          formal_training_resumed=False)
            dump_json(out / 'STATUS.json', status, overwrite=True)
            print(json.dumps(status), flush=True)
    features = np.stack(features)
    for role in ('calibration', 'validation'):
        indices = [i for i, r in enumerate(records) if r['role'] == role]
        rows = [records[i] for i in indices]
        atomic_npz(out / (role + '_records.npz'),
            ids=np.array([r['id'] for r in rows]), task_ids=np.array([r['task_id'] for r in rows]),
            traj_ids=np.array([r['traj_id'] for r in rows]), alignment_verified=np.ones(len(rows), dtype=bool),
            eef_pos_m=np.array([r['eef_pos_m'] for r in rows]),
            eef_quat_xyzw=np.array([r['eef_quat_xyzw'] for r in rows]),
            gripper_open_fraction=np.array([r['gripper_open_fraction'] for r in rows]),
            agent_unmod=features[indices, 0], eih_unmod=features[indices, 1],
            future_actions=np.array([r['future_actions'] for r in rows], dtype=np.float32),
            future_valid=np.array([r['future_valid'] for r in rows], dtype=bool))
    assert not torch.cuda.is_initialized()
    dump_json(out / 'FEATURE_EXPORT_COMPLETE.json', dict(status='PASS', records=len(records), images_checked=images_checked,
        binding=binding, seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        alignment_scope='DIRECT_HDF5_ANCHORS: pixels, pose, gripper and future actions re-read exactly; NOT incremental .pt mapping',
        outputs={role+'_records.npz': sha256(out/(role+'_records.npz')) for role in ('calibration', 'validation')},
        formal_training_resumed=False), overwrite=True)


if __name__ == '__main__':
    main()
