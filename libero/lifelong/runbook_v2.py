"""Protocol orchestration dispatched from the three published Hydra entrypoints.

Models, preprocessing, GMM objective, samplers and selection are the release code.
This module adds provenance, epoch checkpoints, replay transactions and evaluation.
"""
import copy
from functools import partial
import json
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
from easydict import EasyDict
from omegaconf import OmegaConf
from libero.lifelong.runbook_core import (sha256, atomic_json, strip_prefix, rng_state,
    restore_rng, strict_downstream_load, transactional_buffer_update, metrics)
from libero.lifelong.modal_contract import (validate_flags, check_metadata, validate_contract,
    semantic_tests, frontend_hash)
from libero.lifelong.utils import control_seed, get_task_embs, make_linear_schedule_with_warmup


def json_read(path):
    return json.loads(Path(path).read_text())


def append_json(path, value):
    from libero.lifelong.runbook_core import json_default
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, default=json_default, allow_nan=False) + '\n')


def save_torch(path, value):
    temp = str(path) + '.tmp'
    torch.save(value, temp)
    os.replace(temp, str(path))


def metadata(contract):
    return {'modal_contract': contract, 'modal_fingerprint': contract['fingerprint']}


def status(run, state, **kwargs):
    previous = json_read(run / 'RUN_STATUS.json') if (run / 'RUN_STATUS.json').exists() else {}
    previous.update(status=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs)
    atomic_json(run / 'RUN_STATUS.json', previous)


def start(hydra_cfg, phase, distributed=True):
    cfg = EasyDict(OmegaConf.to_container(hydra_cfg, resolve=True))
    validate_flags(cfg)
    assert cfg.seed == 42 and cfg.benchmark_name == 'LIBERO_GOAL'
    assert cfg.gamma == .7 and cfg.past_data_rate == .5
    assert cfg.train.n_epochs == 100 and cfg.eval.n_eval == 20
    assert cfg.data.seq_len == 8 and cfg.data.demo_limit == 10
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    if distributed:
        dist.init_process_group('nccl')
        assert dist.get_world_size() == 1
    torch.cuda.set_device(0)
    run = Path(cfg.run_dir)
    contract = json_read(run / 'modal_contract.json')
    validate_contract(contract, contract)
    assert sha256(run / 'source.patch') == contract['code_patch_sha256']
    assert sha256(cfg.pretrain) == contract['official_checkpoint_sha256']
    cfg.shape_meta = torch.load(cfg.pretrain, map_location='cpu')['cfg']['shape_meta']
    cfg.experiment_dir = str(run)
    cfg.feature_dir = str(run / 'features')
    config_name = 'resolved_%s%s.yaml' % (phase, '_resume' if cfg.get('runbook_resume', False) else '')
    (run / config_name).write_text(OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(cfg)))))
    control_seed(42)
    return cfg, run, contract


class IndexedRawDataset(Dataset):
    def __init__(self, source):
        self.source = source
    def __len__(self):
        return len(self.source)
    def __getitem__(self, i):
        sample = self.source[i]
        sample['_source_index'] = i
        return sample


class IndexedFeatures(Dataset):
    def __init__(self, source):
        self.source = source
        self.file_paths = source.file_paths
    def __len__(self):
        return len(self.source)
    def __getitem__(self, i):
        sample, tid = self.source[i]
        return sample, tid, i


def export_main(hydra_cfg):
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.datasets import get_dataset, SequenceVLDataset, initialize_obs_worker
    from libero.lifelong.algos.basic import Basic
    cfg, run, contract = start(hydra_cfg, 'export')
    assert cfg.train.batch_size == 1 and cfg.aug is True and cfg.training is False
    if (run / 'features').exists() or (run / 'feature_manifest.json').exists():
        raise RuntimeError('New export requires a fresh feature directory; old caches are not relabeled')
    benchmark = get_benchmark('LIBERO_GOAL')(0)
    tokens = get_task_embs(cfg, [t.language for t in benchmark.tasks])
    # Match source reset immediately before policy construction.
    control_seed(42)
    algo = Basic(10, cfg).cuda()
    algo.policy = DDP(algo.policy, device_ids=[0])
    algo.train()
    frozen = frontend_hash(algo.policy.module.state_dict())
    assert frozen == contract['frozen_frontend_sha256']
    manifest = dict(metadata(contract), complete=False, tasks={}, run_id=cfg.run_id,
                    official_checkpoint_sha256=contract['official_checkpoint_sha256'],
                    resolved_export_sha256=sha256(run / 'resolved_export.yaml'))
    status(run, 'EXPORTING', training_epochs_completed=0, formal_episodes_completed=0)
    for task in range(6, 10):
        ds, shape = get_dataset(os.path.join(cfg.folder, benchmark.get_task_demonstration(task)),
                               cfg.data.obs.modality, seq_len=8, demo_limit=10)
        assert ds.n_demos == 10 and set(ds.hdf5_cache) == set(ds.demos)
        vd = SequenceVLDataset(ds, tokens['input_ids'][task], tokens['attention_mask'][task])
        loader = DataLoader(IndexedRawDataset(vd), batch_size=1, sampler=RandomSampler(vd),
                            num_workers=cfg.train.num_workers,
                            worker_init_fn=partial(initialize_obs_worker, obs_modality=cfg.data.obs.modality))
        algo.start_task(task, loader)
        algo.policy.train()
        entries = []
        for index, data in enumerate(loader):
            source_index = data['_source_index'].item()
            with torch.no_grad():
                algo.observe(data, task, index, True, str(run / 'features'))
            path = run / 'features' / str(task) / ('%d.pt' % index)
            payload = torch.load(str(path), map_location='cpu')
            assert list(payload['feature'].shape) == [1, 8, 4, 512]
            assert list(payload['actions'].shape) == [1, 8, 7]
            assert all(torch.isfinite(v).all() for v in payload.values())
            demo = ds._index_to_demo_id[source_index]
            offset = source_index - ds._demo_id_to_start_indices[demo]
            entries.append({'file': str(path.relative_to(run)), 'sha256': sha256(path),
                'bytes': path.stat().st_size, 'source_index': source_index, 'demo_id': demo,
                'window_start': offset, 'export_index': index, 'dtype': str(payload['feature'].dtype),
                'shape': [1, 8, 4, 512]})
            if index % 100 == 0:
                status(run, 'EXPORTING', task=task, exported_current=index + 1, windows_current=len(ds))
                print('EXPORT', task, index + 1, '/', len(ds), flush=True)
        assert len(entries) == len(ds) and len({v['source_index'] for v in entries}) == len(ds)
        manifest['tasks'][str(task)] = {'name': benchmark.get_task(task).name, 'demo_ids': ds.demos,
                                      'windows': len(ds), 'files': entries}
        atomic_json(run / 'feature_manifest.partial.json', manifest)
        ds.close_and_delete_hdf5_handle()
    assert frontend_hash(algo.policy.module.state_dict()) == frozen
    manifest['complete'] = True
    atomic_json(run / 'feature_manifest.json', manifest)
    status(run, 'FEATURES_READY', exported_total=sum(v['windows'] for v in manifest['tasks'].values()))
    dist.destroy_process_group()


def validate_features(run, contract, hash_files=True):
    manifest = json_read(run / 'feature_manifest.json')
    check_metadata(manifest, contract)
    assert manifest['complete'] and set(manifest['tasks']) == {'6', '7', '8', '9'}
    for task in manifest['tasks'].values():
        assert len(task['demo_ids']) == 10 and task['windows'] == len(task['files'])
        for entry in task['files']:
            p = run / entry['file']
            assert p.is_file() and p.stat().st_size == entry['bytes']
            if hash_files:
                assert sha256(p) == entry['sha256'], str(p)
    return manifest


def train_main(hydra_cfg):
    from libero.lifelong.algos.mlr import MLR, build_dataset_for_task, build_ifa_similarity_matrix, parse_path
    cfg, run, contract = start(hydra_cfg, 'adaptation')
    assert cfg.train.batch_size == 10 and cfg.training is True and cfg.aug is False
    feature_manifest = validate_features(run, contract)
    cpdir = run / 'checkpoints'
    cpdir.mkdir(exist_ok=True)
    manifest_path = cpdir / 'manifest.json'
    finals = json_read(manifest_path) if manifest_path.exists() else {}
    rolling = cpdir / 'rolling_epoch_end.pth'
    resume = bool(cfg.get('runbook_resume', False))
    if not resume and (rolling.exists() or finals):
        raise RuntimeError('Full cannot overwrite existing training; use checked resume')
    algo = MLR(1, cfg).cuda()
    algo.policy = DDP(algo.policy, device_ids=[0])
    atomic_json(run / 'training_load_coverage.json', algo.policy.module.load_report)
    state = torch.load(str(rolling), map_location='cpu') if resume and rolling.exists() else None
    if resume and state is None:
        raise RuntimeError('No epoch-end checkpoint to resume')
    if state:
        check_metadata(state, contract)
        assert state['feature_manifest_sha256'] == sha256(run / 'feature_manifest.json')
        algo.policy.load_state_dict(state['state_dict'], strict=True)
    initial = run / 'buffer'
    initial.mkdir(exist_ok=True)
    if list(initial.glob('*.pt')):
        raise RuntimeError('Initial buffer must be empty')
    atomic_json(initial / 'manifest.json', dict(metadata(contract), complete=True, task=None, files=[]))
    previous_buffer = initial
    global_step = state['global_step'] if state else 0
    start_task = state['task'] if state else 6
    for task in range(6, start_task):
        previous_buffer = run / 'buffer_manifests' / ('after_task_%d' % task)
        check_metadata(json_read(previous_buffer / 'manifest.json'), contract)
    parent_sha = contract['official_checkpoint_sha256'] if start_task == 6 else finals[str(start_task-1)]['sha256']
    for task in range(start_task, 10):
        cfg.buffer_dir = str(previous_buffer)
        buffer_manifest = json_read(previous_buffer / 'manifest.json')
        check_metadata(buffer_manifest, contract)
        for entry in buffer_manifest['files']:
            assert sha256(previous_buffer / entry['name']) == entry['sha256']
            assert int(parse_path(str(previous_buffer / entry['name']))) < task
        dataset = IndexedFeatures(build_dataset_for_task(str(run/'features'), task, str(previous_buffer), 42, start=6))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        loader = DataLoader(dataset, batch_size=10, sampler=sampler,
                            num_workers=cfg.train.num_workers, persistent_workers=False)
        info = build_ifa_similarity_matrix(task, str(run/'features'), str(previous_buffer), top_ratio=.5)
        assert max(info['task_ids']) == task
        atomic_json(run / ('ifa_task%d.json' % task), info)
        algo.start_task(task, loader)
        epoch_start = 0
        if state is not None and state['task'] == task:
            algo.optimizer.load_state_dict(state['optimizer'])
            algo.scheduler.load_state_dict(state['scheduler'])
            epoch_start = state['epoch_index'] + 1
            restore_rng(state['rng'])
            assert state['sampler']['seed'] == sampler.seed
        schedule = make_linear_schedule_with_warmup(500, 100 * len(loader))
        np.savetxt(run / ('lr_schedule_task%d.csv' % task),
                   np.array([[i, 1e-4*schedule(i)] for i in range(100*len(loader)+1)]), delimiter=',',
                   header='completed_optimizer_steps,lr_for_next_step', comments='')
        for epoch in range(epoch_start, 100):
            sampler.set_epoch(epoch)
            algo.policy.train()
            started = time.time()
            seen = Counter()
            counts = Counter()
            sums = Counter()
            epoch_records = []
            status(run, 'TRAINING', task=task, epoch_index=epoch, epoch_completed=False,
                   training_epochs_completed=(task-6)*100+epoch, global_step=global_step)
            for index, data in enumerate(loader):
                ids = list(data[1])
                seen.update(int(x) for x in data[2])
                counts.update(int(x) for x in ids)
                algo.observe(data, task, epoch, ifa_info=info)
                global_step += 1
                row = dict(algo.last_loss_log, task=task, epoch_index=epoch,
                           batch_index=index, global_step=global_step)
                assert all(np.isfinite(row[k]) for k in ('bc','ifa_raw','ifa_weighted','total','grad_norm'))
                row.update(new_samples=ids.count(str(task)), old_samples=len(ids)-ids.count(str(task)))
                epoch_records.append(row)
                for key in ('bc','ifa_raw','ifa_weighted','total'):
                    sums[key] += row[key] * len(ids)
            assert len(seen) == len(dataset) and all(v == 1 for v in seen.values())
            assert sum(counts.values()) == len(dataset)
            old = sum(n for t, n in counts.items() if t != task)
            summary = {'kind': 'epoch', 'task': task, 'epoch_index': epoch,
                'optimizer_steps': len(loader), 'global_step': global_step,
                'new_samples': counts[task], 'old_samples': old, 'repeated_samples': 0,
                'old_fraction': old / len(dataset), 'per_task_counts': dict(counts),
                'duration_seconds': time.time()-started,
                'lr_next': algo.optimizer.param_groups[0]['lr'],
                **{k: v / len(dataset) for k, v in sums.items()}}
            # Epoch logs are committed atomically; resume can overwrite a partial epoch log.
            atomic_json(run / 'logs/epochs' / ('task%d_epoch%03d.json' % (task, epoch)),
                        {'summary': summary, 'steps': epoch_records})
            payload = dict(metadata(contract), state_dict=algo.policy.state_dict(),
                optimizer=algo.optimizer.state_dict(), scheduler=algo.scheduler.state_dict(),
                rng=rng_state(), task=task, epoch_index=epoch, global_step=global_step,
                sampler={'epoch': epoch, 'seed': sampler.seed, 'num_replicas': 1, 'rank': 0},
                completed_tasks=list(range(6, task)), cfg=cfg, parent_checkpoint_sha256=parent_sha,
                feature_manifest=feature_manifest, feature_manifest_sha256=sha256(run/'feature_manifest.json'),
                buffer_manifest=buffer_manifest, buffer_path=str(previous_buffer), run_id=cfg.run_id)
            save_torch(rolling, payload)
            state = payload
            append_json(run / 'logs/train.jsonl', summary)
            status(run, 'TRAINING', task=task, epoch_index=epoch, epoch_completed=True,
                   training_epochs_completed=(task-6)*100+epoch+1, global_step=global_step)
            print('EPOCH_COMPLETE', task, epoch, 'steps', global_step, 'seconds', round(summary['duration_seconds'],2), flush=True)
        assert state['task'] == task and state['epoch_index'] == 99
        assert algo.scheduler.last_epoch == 100 * len(loader)
        assert algo.optimizer.param_groups[0]['lr'] == 0.0
        final_path = cpdir / ('final_task%d_epoch99.pth' % task)
        if str(task) not in finals:
            save_torch(final_path, state)
            loaded = torch.load(str(final_path), map_location='cpu')
            check_metadata(loaded, contract)
            strict_downstream_load(algo.policy.module, loaded['state_dict'])
            finals[str(task)] = dict(metadata(contract), path=str(final_path), sha256=sha256(final_path),
                task=task, epoch_index=99, parent_checkpoint_sha256=parent_sha,
                global_step=global_step, load_coverage=1.0)
            atomic_json(manifest_path, finals)
            del loaded
        else:
            assert sha256(final_path) == finals[str(task)]['sha256']
        buffer = transactional_buffer_update(run/'features', task, previous_buffer,
                                             run/'buffer_manifests', modal_contract=contract)
        check_metadata(buffer, contract)
        previous_buffer = Path(buffer['generation'])
        parent_sha = finals[str(task)]['sha256']
        state = None
    status(run, 'TRAINING_COMPLETED_EVAL_PENDING', training_epochs_completed=400,
           formal_episodes_completed=0, global_step=global_step, final_checkpoints=finals)
    dist.destroy_process_group()


def eval_main(hydra_cfg):
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.lifelong.models.complete_policy import CompletePolicy
    from libero.lifelong.models.brief import BriefPolicy
    import robomimic.utils.obs_utils as ObsUtils
    cfg, run, contract = start(hydra_cfg, 'eval', distributed=False)
    finals = json_read(run / 'checkpoints/manifest.json')
    if set(finals) != {'6','7','8','9'}:
        raise ValueError('Four final checkpoints required; no waiting or skipping')
    benchmark = get_benchmark('LIBERO_GOAL')(0)
    tokens = get_task_embs(cfg, [t.language for t in benchmark.tasks])
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': cfg.data.obs.modality})
    out = run / 'evaluation'
    out.mkdir(exist_ok=True)
    all_records = []
    for stage in range(6,10):
        meta = finals[str(stage)]
        check_metadata(meta, contract)
        assert meta['epoch_index'] == 99 and sha256(meta['path']) == meta['sha256']
        policy = CompletePolicy(cfg, cfg.shape_meta).cuda().eval()
        checkpoint = torch.load(meta['path'], map_location='cpu')
        check_metadata(checkpoint, contract)
        load_report = strict_downstream_load(policy, checkpoint['state_dict'])
        assert frontend_hash(policy.state_dict()) == contract['frozen_frontend_sha256']
        atomic_json(out / ('load_stage%d.json' % stage), load_report)
        # Test the actual online state branch after replacing downstream parameters.
        brief_cfg = copy.deepcopy(cfg)
        brief_cfg.aug = False
        brief_cfg.training = True
        brief = BriefPolicy(brief_cfg, cfg.shape_meta).cuda().eval()
        strict_downstream_load(brief, checkpoint['state_dict'])
        del checkpoint
        for task_id in range(6,stage+1):
            task = benchmark.get_task(task_id)
            init_path = Path(cfg.init_states_folder) / task.problem_folder / task.init_states_file
            source_row = json_read(run/'data_manifest.json')['tasks'][task_id]
            assert sha256(init_path) == source_row['init_sha256']
            assert sha256(Path(cfg.bddl_folder)/task.problem_folder/task.bddl_file) == source_row['bddl_sha256']
            init_states = torch.load(str(init_path), map_location='cpu')
            assert len(init_states) >= 20
            import hashlib
            init_hashes = [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in init_states[:20]]
            assert len(set(init_hashes)) == 20
            cell_path = out / ('stage%d_task%d.json' % (stage, task_id))
            if cell_path.exists():
                cell = json_read(cell_path)
                check_metadata(cell, contract)
                assert cell['checkpoint_sha'] == meta['sha256'] and len(cell['episodes']) == 20
                all_records.extend(cell['episodes'])
                continue
            control_seed(42)
            records = []
            env_args = {'bddl_file_name': os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
                        'camera_heights': 128, 'camera_widths': 128}
            env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(10)])
            try:
                for batch in range(2):
                    seed = 42 + batch
                    env.seed(seed)
                    env.reset()
                    env.seed(seed)
                    obs = env.set_init_state(init_states[batch*10:(batch+1)*10])
                    policy.reset()
                    history = []
                    for _ in range(5):
                        obs, _, _, _ = env.step(np.zeros((10,7)))
                    dones = [False]*10
                    terminal_steps = [None]*10
                    terminal_times = [None]*10
                    began = time.time()
                    for steps in range(1,601):
                        data = raw_obs_to_tensor_obs(obs, tokens['input_ids'][task_id], tokens['attention_mask'][task_id], cfg)
                        history.append(data)
                        history = history[-8:]
                        batch_data = {'task_emb': data['task_emb'], 'masks': data['masks'], 'obs': {
                            key: torch.stack([x['obs'][key] for x in history], 1) for key in data['obs']}}
                        if batch == 0 and steps == 1:
                            semantic_tests(policy, brief, policy.preprocess_input(copy.deepcopy(batch_data), False),
                                           out / ('online_modality_stage%d_task%d.json' % (stage,task_id)))
                        actions = policy.get_action(batch_data)
                        assert actions.shape == (10,7) and np.isfinite(actions).all()
                        obs, _, done, _ = env.step(actions)
                        for worker in range(10):
                            if bool(done[worker]) and not dones[worker]:
                                terminal_steps[worker] = steps
                                terminal_times[worker] = time.time()-began
                            dones[worker] = dones[worker] or bool(done[worker])
                        if all(dones):
                            break
                    for worker in range(10):
                        init_id = batch*10+worker
                        row = {'kind': 'FORMAL', 'run_id': cfg.run_id, 'training_seed': 42,
                            'stage': stage, 'task_id': task_id, 'task_name': task.name,
                            'checkpoint_sha': meta['sha256'], 'eval_seed': 42,
                            'env_batch_seed': seed, 'worker_seed': seed+worker,
                            'init_state_id': init_id, 'init_state_hash': init_hashes[init_id],
                            'success': bool(dones[worker]), 'policy_steps': terminal_steps[worker] or steps,
                            'termination_reason': 'success' if dones[worker] else 'max_policy_steps',
                            'duration_seconds': terminal_times[worker] or time.time()-began,
                            'exception': None, 'modal_fingerprint': contract['fingerprint']}
                        records.append(row)
                        append_json(out/'episode_journal.jsonl', row)
                    print('EVAL_BATCH', stage, task_id, batch, 'success', sum(dones), flush=True)
            except Exception as exc:
                append_json(out/'exceptions.jsonl', {'stage': stage,'task': task_id,'batch': batch,
                    'initial_state_ids': list(range(batch*10,(batch+1)*10)), 'exception': repr(exc),
                    'retry_rule': 'Abort current cell. Explicit resume restarts same cell and initial states from RNG42; no substitutes.'})
                raise
            finally:
                env.close()
            assert len(records) == 20
            atomic_json(cell_path, dict(metadata(contract), checkpoint_sha=meta['sha256'], episodes=records))
            all_records.extend(records)
            status(run, 'EVALUATING', formal_episodes_completed=len(all_records), current_stage=stage, current_task=task_id)
        del policy, brief
        torch.cuda.empty_cache()
    assert len(all_records) == 200
    temp = out/'rollouts.jsonl.tmp'
    with temp.open('w') as f:
        for row in all_records:
            f.write(json.dumps(row)+'\n')
    os.replace(str(temp), str(out/'rollouts.jsonl'))
    matrix = np.full((4,4),np.nan)
    for stage in range(6,10):
        for task in range(6,stage+1):
            rows = [r for r in all_records if r['stage']==stage and r['task_id']==task]
            assert len(rows)==20 and {r['init_state_id'] for r in rows} == set(range(20))
            matrix[stage-6,task-6] = sum(r['success'] for r in rows)/20
    np.savetxt(out/'success_matrix.csv', matrix, delimiter=',',header='task6,task7,task8,task9',comments='')
    atomic_json(run/'metrics.json',metrics(matrix))
    status(run, 'COMPLETED', training_epochs_completed=400, formal_episodes_completed=200)
