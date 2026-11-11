"""Read-only historical inputs and isolated evaluation output bookkeeping."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = '/data/mlr_ifa/env/bin/python'
ENV_LOCK = Path('/data/mlr_ifa/locks/reverse_main_protocol_environment.lock')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen():
    contract = read(ROOT / 'CONTRACT.json')
    for name, expected in contract['files'].items():
        assert sha(ROOT / name) == expected, name
    return read(ROOT / 'SPEC.json')


def verify_inputs(case):
    origin = Path(case['reference_dir'])
    assert sha(origin / 'CONTRACT.json') == case['reference_contract_sha256']
    assert sha(origin / 'evaluate.py') == sha(ROOT / 'evaluate.py')
    config_root = Path(case['config_root'])
    config = read(config_root / 'resolved_eval.json')
    assert config == case['eval_config'] and config['data']['task_order_index'] == 0
    assert sha(config['pretrain']) == case['official_sha256']
    for name, expected in read(config_root / 'CONTRACT.json')['source_sha256'].items():
        assert sha(Path(case['source_root']) / name) == expected, name
    for model in case['models'].values():
        assert Path(model['path']).stat().st_size == model['bytes']
        assert sha(model['path']) == model['sha256'], model['path']


def capacity():
    root = os.statvfs('/')
    data = os.statvfs('/data')
    values = dict(root_bytes=root.f_bavail * root.f_frsize, data_bytes=data.f_bavail * data.f_frsize)
    assert values['root_bytes'] >= 1024**3
    assert values['data_bytes'] >= 2 * 1024**3 + 64 * 1024**2
    return values


def processes(directory):
    path = str(directory)
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cwd = os.readlink(str(proc / 'cwd'))
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if cwd == path or cwd.startswith(path + '/') or any(part.startswith((path + '/').encode()) for part in cmd):
                rows.append(int(proc.name))
        except OSError:
            continue
    return rows


def environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', LIBERO_CONFIG_PATH='/data/mlr_ifa/config',
               HF_HOME='/data/mlr_ifa/cache/huggingface', MPLBACKEND='Agg', CAET_VARIANT='ALIGN')
    return env
