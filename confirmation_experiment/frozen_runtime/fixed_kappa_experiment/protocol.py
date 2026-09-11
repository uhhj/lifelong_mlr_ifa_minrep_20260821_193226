"""Read-only recipe/provenance checks for reusing the two completed endpoints."""
import hashlib
import json
from pathlib import Path

from metrics import read_fixed_matrix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def endpoint_dirs(repo, seed):
    return {
        'mlr': repo / 'runs/final_paper_text_faithful/mlr' / ('seed%d' % seed),
        'hard': repo / 'runs/replay_constraint/aggregate_v1/formal/constraint' / ('seed%d' % seed),
    }


def common_recipe(cfg):
    return {k: cfg[k] for k in ('seed', 'benchmark_name', 'data', 'policy', 'train',
        'start_task', 'end_task', 'aug', 'past_data_rate', 'ifa_enabled', 'repro_scheduler_mode',
        'task_embedding_format', 'save_final_checkpoint', 'load_previous_model')}


def check_fixed_recipe(cfg, seed, smoke=False):
    expected = {'seed': seed, 'benchmark_name': 'LIBERO_OBJECT', 'start_task': 6,
        'end_task': 8 if smoke else 10, 'aug': False, 'past_data_rate': .5,
        'ifa_enabled': False, 'repro_scheduler_mode': 'batch_step',
        'save_final_checkpoint': True, 'load_previous_model': False}
    for key, value in expected.items():
        require(cfg[key] == value, 'Unexpected recipe: ' + key)
    train = cfg['train']
    for key, value in {'n_epochs': 1 if smoke else 100, 'batch_size': 10, 'num_workers': 0,
                       'grad_clip': 100., 'loss_scale': 1., 'resume': False}.items():
        require(train[key] == value, 'Unexpected train recipe: ' + key)
    require(train['optimizer'] == {'name': 'torch.optim.AdamW', 'kwargs': {
        'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .1}}, 'Optimizer changed')
    require(cfg['data']['seq_len'] == 8 and cfg['data']['demo_limit'] == 10, 'Data recipe changed')


def audit_references(repo, seed, features, checkpoint):
    refs = {}
    configs = []
    for arm, directory in endpoint_dirs(repo, seed).items():
        config_path = directory / 'run_001/config.json'
        log = directory / 'primary_ep99.log'
        cfg = read(config_path)
        check_fixed_recipe(cfg, seed)
        require(Path(cfg['feature_dir']).resolve() == features.resolve(), 'Endpoint feature path differs')
        require(Path(cfg['pretrain']).resolve() == checkpoint.resolve(), 'Endpoint initial checkpoint differs')
        if arm == 'hard':
            require(cfg['lifelong']['algo'] == 'MLRReplayConstraint', 'Wrong hard endpoint algorithm')
            require(cfg['lifelong']['projection_enabled'] is True, 'Hard endpoint disabled')
        else:
            require(cfg['lifelong']['algo'].lower() == 'mlr', 'Wrong MLR endpoint algorithm')
        refs[arm] = {'directory': str(directory), 'config_sha256': sha(config_path),
                     'log_sha256': sha(log), 'matrix': read_fixed_matrix(log)}
        configs.append(cfg)
    require(common_recipe(configs[0]) == common_recipe(configs[1]), 'Endpoint common recipes differ')
    return {'endpoints': refs, 'common_recipe': common_recipe(configs[0]),
        'initial_checkpoint_sha256': sha(checkpoint),
        'limitation': 'Existing endpoints identify cache/checkpoint paths; historical cache content hashes were not recorded.'}


def source_hashes(repo):
    paths = ['libero/lifelong/algos/fixed_kappa.py', 'libero/lifelong/algos/mlr_fixed_kappa.py',
        'libero/configs/lifelong/mlr_fixed_kappa.yaml', 'libero/lifelong/algos/mlr.py',
        'libero/lifelong/algos/base.py', 'libero/lifelong/models/brief.py',
        'libero/lifelong/models/base_policy.py', 'libero/lifelong/models/policy_head.py',
        'libero/lifelong/lifelong_adaptation.py', 'libero/lifelong/test.py']
    return {p: sha(repo / p) for p in paths}
