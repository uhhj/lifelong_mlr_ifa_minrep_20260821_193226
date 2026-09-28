"""Resource and bounded real-data smoke helpers; invoked by goal_v2_driver."""
import argparse
import copy
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from libero.lifelong.runbook_core import (sha256, atomic_json, strip_prefix,
    strict_downstream_load, ifa_loss, metrics, rng_state, restore_rng)


def config(args):
    with initialize_config_dir(config_dir=str(ROOT / 'libero/configs'), version_base=None):
        cfg = compose(config_name='config_adaption', overrides=[
            'benchmark_name=LIBERO_GOAL', 'policy=complete_policy', 'lifelong=mlr',
            'seed=42', 'train.batch_size=10', 'train.n_epochs=100', 'eval.n_eval=20',
            'training=false', '+gamma=0.7', '+past_data_rate=0.5', '+aug=true',
            '+start_task=6', '+end_task=10'])
    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.pretrain = args.pretrained
    cfg.folder = args.data_root
    cfg.bddl_folder = args.bddl_root
    cfg.init_states_folder = args.init_states_root
    cfg.experiment_dir = str(args.run_dir / 'smoke')
    cfg.feature_dir = str(args.run_dir / 'smoke/features')
    cfg.buffer_dir = str(args.run_dir / 'smoke/buffer')
    return cfg


def preflight(args, cfg):
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.datasets import get_dataset
    report = {'platform': platform.platform(), 'python': sys.version,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'egl_device': os.environ.get('MUJOCO_EGL_DEVICE_ID'),
              'source': str(ROOT), 'source_commit': subprocess.check_output(
                  ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    assert report['source_commit'] == 'a721807f2f7fa7f9833dcc85c592d78e9accd00b'
    for name in ['torch', 'torchvision', 'numpy', 'h5py', 'mujoco', 'robosuite',
                 'robomimic', 'transformers', 'peft', 'hydra']:
        report[name] = getattr(importlib.import_module(name), '__version__', 'imported')
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    assert torch.cuda.device_count() == 1
    report['gpu'] = torch.cuda.get_device_name(0)
    report['torch_cuda'] = torch.version.cuda
    report['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    probe = torch.ones((32, 32), device='cuda')
    assert torch.isfinite(probe @ probe).all()
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    (args.run_dir / 'requirements.lock.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    atomic_json(args.run_dir / 'environment.json', report)
    ckpt = torch.load(args.pretrained, map_location='cpu')
    assert ckpt['cfg']['benchmark_name'] == 'LIBERO_GOAL'
    assert 'libero_goal' in ckpt['cfg']['experiment_dir'].lower()
    checkpoint_hash = sha256(args.pretrained)
    assert checkpoint_hash == '241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7'
    state = strip_prefix(ckpt['state_dict'])
    atomic_json(args.run_dir / 'pretrained_manifest.json', {
        'path': args.pretrained, 'sha256': checkpoint_hash,
        'bytes': Path(args.pretrained).stat().st_size,
        'source': 'https://drive.google.com/drive/folders/1NOiyhkb1Hni_ElhqcN7GkiXnfCqOKiSH',
        'checkpoint_cfg': ckpt['cfg'], 'metadata_seed': ckpt['cfg'].get('seed'),
        'seed_caveat': 'Metadata records 42; unpublished pretraining trajectory not independently verified.',
        'key_shapes': {k: list(v.shape) for k, v in state.items()},
        'use_state_in_checkpoint_cfg': ckpt['cfg']['policy'].get('use_state', 'ABSENT')})
    benchmark = get_benchmark('LIBERO_GOAL')(0)
    manifest = {'benchmark': 'LIBERO_GOAL', 'task_order_index': 0, 'tasks': []}
    for tid in range(10):
        task = benchmark.get_task(tid)
        data = Path(args.data_root) / benchmark.get_task_demonstration(tid)
        bddl = Path(args.bddl_root) / task.problem_folder / task.bddl_file
        init = Path(args.init_states_root) / task.problem_folder / task.init_states_file
        for p in [data, bddl, init]:
            assert p.is_file(), str(p)
        states = torch.load(str(init), map_location='cpu')
        assert len(states) >= 20
        state_hashes = [__import__('hashlib').sha256(np.asarray(v).tobytes()).hexdigest()
                        for v in states[:20]]
        assert len(set(state_hashes)) == 20
        row = {'task_id': tid, 'name': task.name, 'language': task.language,
               'group': 'pretrained' if tid < 6 else 'incremental',
               'data': str(data), 'data_sha256': sha256(data), 'data_bytes': data.stat().st_size,
               'bddl': str(bddl), 'bddl_sha256': sha256(bddl),
               'init_states': str(init), 'init_sha256': sha256(init),
               'init_count': len(states), 'first20_init_hashes': state_hashes}
        if tid >= 6:
            ds, shape = get_dataset(str(data), cfg.data.obs.modality, seq_len=8, demo_limit=10)
            assert ds.n_demos == 10 and len(ds) == sum(ds._demo_id_to_demo_length.values())
            assert set(ds._index_to_demo_id.values()) == set(ds.demos)
            assert set(ds.hdf5_cache) == set(ds.demos)
            # Inspect first and last window of every selected demo and action padding.
            for demo in ds.demos:
                start = ds._demo_id_to_start_indices[demo]
                end = start + ds._demo_id_to_demo_length[demo] - 1
                for index in [start, end]:
                    sample = ds[index]
                    assert sample['actions'].shape == (8, 7)
                    assert np.isfinite(sample['actions']).all()
                target = ds.hdf5_file['data'][demo]['actions'][-1]
                np.testing.assert_allclose(ds[end]['actions'], np.tile(target, (8, 1)))
            row.update({'demo_ids': ds.demos, 'windows': len(ds),
                        'demo_lengths': ds._demo_id_to_demo_length, 'cache_demo_ids': list(ds.hdf5_cache)})
            ds.close_and_delete_hdf5_handle()
            del ds
        manifest['tasks'].append(row)
        print('AUDITED_TASK', tid, task.name, row.get('windows'), flush=True)
    atomic_json(args.run_dir / 'data_manifest.json', manifest)
    atomic_json(args.run_dir / 'preflight_result.json', {'status': 'PASS', 'formal_status': 'NOT_STARTED'})
    # Dry-run the full per-stage optimizer-step schedule without doing optimization.
    from libero.lifelong.utils import make_linear_schedule_with_warmup
    past = 0
    schedule_manifest = []
    for row in manifest['tasks'][6:]:
        n = row['windows'] + past
        batches = (n + 9) // 10
        total = 100 * batches
        schedule = make_linear_schedule_with_warmup(500, total)
        vals = np.array([1e-4 * schedule(i) for i in range(total + 1)])
        assert vals[0] == 0 and vals[500] == 1e-4 and vals[-1] == 0
        assert np.all(np.diff(vals[500:]) < 0)
        np.savetxt(args.run_dir / ('lr_schedule_task%d.csv' % row['task_id']),
                   np.column_stack([np.arange(total + 1), vals]), delimiter=',',
                   header='completed_optimizer_steps,lr_for_next_step', comments='')
        schedule_manifest.append({'task': row['task_id'], 'new_windows': row['windows'],
                                  'old_windows': past, 'batches_per_epoch': batches, 'total_steps': total})
        past += (row['windows'] + 1) // 2
    atomic_json(args.run_dir / 'lr_schedule_manifest.json', schedule_manifest)


def software_tests(out):
    from libero.lifelong.algos.mlr import _pairwise_cosine_mean
    from libero.lifelong.utils import _strip_module_prefix1
    results = []
    value = torch.tensor([3.0])
    for state in [{'x': value}, {'module.x': value}]:
        assert torch.equal(_strip_module_prefix1(state)['x'], value)
    try:
        _strip_module_prefix1({'x': value, 'module.y': value})
        raise AssertionError('Mixed-prefix checkpoint accepted')
    except ValueError:
        pass
    results.append('prefix_with_without_and_mixed')
    r = np.array([[.8, np.nan, np.nan, np.nan], [.7, .9, np.nan, np.nan],
                  [.6, .8, .7, np.nan], [.5, .7, .6, .8]])
    m = metrics(r)
    np.testing.assert_allclose([m[k] for k in ['FWT', 'NBT', 'AUC', 'FinalAvgSR']],
                               [.8, .15, .725, .65])
    r[np.tril_indices(4)] = [.2, .3, .2, .4, .3, .2, .5, .4, .3, .2]
    assert metrics(r)['NBT'] < 0
    r[3, 0] = np.nan
    try:
        metrics(r)
        raise AssertionError('Incomplete metrics accepted')
    except ValueError:
        pass
    results.append('metrics_contract_negative_NBT_and_incomplete_rejection')
    torch.manual_seed(42)
    a, b = torch.randn(17, 512), torch.randn(23, 512)
    dense = (torch.nn.functional.normalize(a, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).T).mean()
    err = abs(_pairwise_cosine_mean(a, b) - dense.item())
    assert err < 1e-7
    results.append('exact_similarity_identity_error_%g' % err)
    features = torch.zeros(3, 8, 4, 512)
    features[0, :, 0, 1] = 1  # old sample deliberately first
    features[1, :, 0, 0] = 1
    features[2, :, 0, 2] = 1
    z = features[:, 0, 0].clone().requires_grad_(True)
    info = {'task_ids': [6, 7, 8], 'selected_pair_mask': torch.tensor(
        [[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=torch.bool), 'sim_language': torch.eye(3)}
    neg = {6: features[0, 0, 0], 7: -features[0, 0, 0]}
    correct = ifa_loss(z, features, [6, 8, 8], 8, info, neg, .7)
    bad_features = features.clone()
    bad_features[1:, 0, 0] = features[0, 0, 0]
    old_bug = ifa_loss(z, bad_features, [6, 8, 8], 8, info, neg, .7)
    assert old_bug > correct + .1
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(correct, ifa_loss(z[perm], features[perm], [8, 6, 8], 8, info, neg, .7))
    info1, info2 = copy.deepcopy(info), copy.deepcopy(info)
    info1['selected_pair_mask'][2, 1] = False
    info2['selected_pair_mask'][2, 0] = False
    vals = [ifa_loss(z, bad_features, [6, 8, 8], 8, v, neg, .7) for v in [info1, info2]]
    assert torch.allclose(old_bug, (vals[0] + vals[1]) / 2)
    assert ifa_loss(z, features, [6, 6, 6], 8, info, neg, .7).item() == 0
    assert ifa_loss(z, features, [6, 8, 8], 8, None, {}, .7).item() == 0
    empty_info = copy.deepcopy(info)
    empty_info['selected_pair_mask'].zero_()
    assert ifa_loss(z, features, [6, 8, 8], 8, empty_info, {}, .7).item() == 0
    old_bug.backward()
    assert torch.isfinite(z.grad).all() and torch.count_nonzero(z.grad[0]) == 0
    results.append('IFA_old_first_own_positive_permutation_pair_mean_empty_and_gradients')
    atomic_json(out / 'software_tests.json', {'status': 'PASS', 'tests': results,
                                             'synthetic_only': True})


def model_hash(model, frozen_only=False):
    import hashlib
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not frozen_only or not name.startswith(('temporal_transformer.', 'policy_head.')):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def smoke(args, cfg):
    from torch.utils.data import DataLoader, RandomSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    import robomimic.utils.tensor_utils as TU
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.datasets import get_dataset, SequenceVLDataset
    from libero.lifelong.models.complete_policy import CompletePolicy
    from libero.lifelong.algos.mlr import MLR
    from libero.lifelong.utils import control_seed, get_task_embs
    out = args.run_dir / 'smoke'
    out.mkdir(exist_ok=False)
    software_tests(out)
    control_seed(42)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 1
    torch.cuda.set_device(0)
    benchmark = get_benchmark('LIBERO_GOAL')(0)
    tokens = get_task_embs(cfg, [t.language for t in benchmark.tasks])
    ds, shape = get_dataset(str(Path(cfg.folder) / benchmark.get_task_demonstration(6)),
                             cfg.data.obs.modality, seq_len=8, demo_limit=10)
    cfg.shape_meta = shape
    vd = SequenceVLDataset(ds, tokens['input_ids'][6], tokens['attention_mask'][6])
    loader = DataLoader(vd, batch_size=1, sampler=RandomSampler(vd), num_workers=0)
    full = CompletePolicy(cfg, shape).cuda().train()
    frozen_hash = model_hash(full, True)
    expected_trainable = [n for n, p in full.named_parameters() if p.requires_grad]
    assert all(n.startswith(('temporal_transformer.', 'policy_head.')) for n in expected_trainable)
    result = {'mode': 'SMOKE_ONLY', 'effective_use_state': cfg.policy.use_state,
              'spatial_frozen_before': frozen_hash, 'export_model_requires_grad_names': expected_trainable,
              'export_mode': 'train; original augmentation enabled; no optimizer updates',
              'full_strict_official_load': {'missing': [], 'unexpected': [], 'coverage': 1.0,
                                           'keys': len(full.state_dict())}}
    windows = []
    feature_dir = out / 'features/6'
    feature_dir.mkdir(parents=True)
    for i, batch in enumerate(loader):
        if i == 2:
            break
        batch = TU.map_tensor(batch, lambda x: x.cuda())
        with torch.no_grad():
            processed = full.preprocess_input(copy.deepcopy(batch))
            latent = full(processed)
        assert list(latent.shape) == [1, 8, 4, 512]
        assert torch.isfinite(latent).all() and torch.isfinite(batch['actions']).all()
        torch.save({'feature': latent.cpu(), 'actions': batch['actions'].cpu()}, str(feature_dir / ('%d.pt' % i)))
        windows.append({'feature': latent.detach().unsqueeze(0), 'actions': batch['actions'].unsqueeze(0)})
    # Same preprocessed, deterministic input through both decoder interfaces.
    full.eval()
    cfg_train = copy.deepcopy(cfg)
    cfg_train.training = True
    cfg_train.aug = False
    cfg_train.policy.policy_type = 'BriefPolicy'
    algo = MLR(1, cfg_train).cuda()
    assert all(p.requires_grad for p in algo.policy.parameters())
    result['brief_official_load'] = algo.policy.load_report
    algo.policy.eval()
    with torch.no_grad():
        deterministic = full.spatial_encode(processed)
        torch.testing.assert_close(full.temporal_encode(deterministic), algo.policy.temporal_encode(deterministic))
    from libero.lifelong.modal_contract import semantic_tests
    result['modal_semantics'] = semantic_tests(full, algo.policy, processed, args.run_dir / 'modality_tests.json')
    algo.policy = DDP(algo.policy, device_ids=[0])
    algo.start_task(6, [0] * 20)
    algo.policy.train()
    initial_hash = model_hash(algo.policy)
    logs = []
    for i in range(2):
        loss = algo.observe([windows[i], ['6']], 6, 0, ifa_info=None)
        assert np.isfinite(loss)
        assert np.isfinite(algo.last_loss_log['grad_norm'])
        logs.append(dict(algo.last_loss_log))
    assert logs[0]['lr_used'] == 0 and logs[1]['lr_used'] > 0
    assert model_hash(algo.policy) != initial_hash
    assert model_hash(full, True) == frozen_hash
    result['optimizer_steps'] = logs
    result['spatial_frozen_after'] = model_hash(full, True)
    checkpoint = out / 'epoch_boundary_smoke.pth'
    torch.save({'state_dict': algo.policy.state_dict(), 'optimizer': algo.optimizer.state_dict(),
                'scheduler': algo.scheduler.state_dict(), 'rng': rng_state(), 'epoch_index': 0,
                'global_step': 2, 'sampler_epoch': 0, 'cfg': cfg_train, 'completed_tasks': [],
                'features': [sha256(p) for p in sorted(feature_dir.glob('*.pt'))], 'buffer': []}, str(checkpoint))
    algo.observe([windows[0], ['6']], 6, 1, ifa_info=None)
    continuous_hash = model_hash(algo.policy)
    saved = torch.load(str(checkpoint), map_location='cpu')
    algo.policy.load_state_dict(saved['state_dict'], strict=True)
    algo.optimizer.load_state_dict(saved['optimizer'])
    algo.scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    algo.observe([windows[0], ['6']], 6, 1, ifa_info=None)
    assert model_hash(algo.policy) == continuous_hash
    result['epoch_boundary_smoke_resume_exact_next_update'] = True
    full_before = model_hash(full, True)
    result['stage_downstream_load_test'] = strict_downstream_load(full, saved['state_dict'])
    assert model_hash(full, True) == full_before
    bad = dict(saved['state_dict'])
    bad.pop(next(iter(bad)))
    try:
        strict_downstream_load(full, bad)
        raise AssertionError('Missing trainable tensor accepted')
    except ValueError:
        pass
    # Closed loop starts from the genuine official model, not smoke-trained weights.
    full.load_state_dict(strip_prefix(torch.load(cfg.pretrain, map_location='cpu')['state_dict']), strict=True)
    del algo, saved, bad, windows
    torch.cuda.empty_cache()
    atomic_json(out / 'model_tests.json', result)
    ds.close_and_delete_hdf5_handle()
    closed_loop_smoke(args, cfg, full, benchmark, tokens)
    atomic_json(out / 'result.json', {'status': 'PASS', 'formal_status': 'NOT_STARTED',
                                     'training_epochs_completed': 0, 'formal_rollouts': 0})
    dist.destroy_process_group()


def closed_loop_smoke(args, cfg, policy, benchmark, tokens):
    from libero.libero.envs import OffScreenRenderEnv
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.lifelong.utils import control_seed
    task = benchmark.get_task(6)
    control_seed(42)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(cfg.bddl_folder) / task.problem_folder / task.bddl_file),
                            camera_heights=128, camera_widths=128)
    states = torch.load(str(Path(cfg.init_states_folder) / task.problem_folder / task.init_states_file))
    start = time.time()
    record = {'kind': 'SMOKE_ONLY', 'task_id': 6, 'task_name': task.name, 'seed': 42,
              'init_state_id': 0, 'checkpoint_sha256': sha256(cfg.pretrain), 'success': False,
              'policy_steps': 0, 'max_steps': 600, 'exception': None}
    try:
        env.seed(42)
        env.reset()
        env.seed(42)
        obs = env.set_init_state(states[0])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))
        history = []
        policy.reset()
        for step in range(1, 601):
            data = raw_obs_to_tensor_obs([obs], tokens['input_ids'][6], tokens['attention_mask'][6], cfg)
            history.append(data)
            history = history[-8:]
            batch = {'task_emb': data['task_emb'], 'masks': data['masks'], 'obs': {
                key: torch.stack([x['obs'][key] for x in history], 1) for key in data['obs']}}
            actions = policy.get_action(batch)
            assert actions.shape == (1, 7) and np.isfinite(actions).all()
            obs, _, done, _ = env.step(actions[0])
            actual_success = bool(env.check_success())
            assert bool(done) == actual_success
            record['policy_steps'] = step
            if step % 100 == 0:
                print('SMOKE_ROLLOUT_STEP', step, flush=True)
            if actual_success:
                record['success'] = True
                break
        record['termination_reason'] = 'success' if record['success'] else 'max_policy_steps'
    except Exception as exc:
        record['exception'] = repr(exc)
        raise
    finally:
        env.close()
        record['duration_seconds'] = time.time() - start
        atomic_json(args.run_dir / 'smoke/rollout.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'smoke', 'full', 'resume'], required=True)
    for name in ['pretrained', 'data-root', 'bddl-root', 'init-states-root']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = config(args)
    if args.dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(cfg)))))
        print('FORMAL STATUS: BLOCKED; user deferred use_state decision. No computation performed.')
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(cfg)))))
    atomic_json(args.run_dir / 'RUN_STATUS.json', {'status': 'BLOCKED',
        'reason': 'User explicitly deferred use_state scientific protocol choice',
        'formal_epochs_completed': 0, 'formal_episodes_completed': 0,
        'stage_final_checkpoints': [], 'active_mode': args.mode})
    if args.mode in ['full', 'resume']:
        raise SystemExit('BLOCKED: use_state decision unresolved; formal orchestration not released.')
    if args.mode == 'preflight':
        preflight(args, cfg)
    else:
        smoke(args, cfg)


if __name__ == '__main__':
    raise SystemExit('Use scripts/run_goal_seed42.sh for the state-v2 protocol')
