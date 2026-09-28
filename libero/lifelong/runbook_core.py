"""Audited helpers for the user-registered GOAL seed42 protocol."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default))
    os.replace(str(temp), str(path))


def strip_prefix(state):
    if not state:
        return state
    prefixed = [k.startswith('module.') for k in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError('Mixed module prefixes')
    return {k[7:] if all(prefixed) else k: v for k, v in state.items()}


def strict_downstream_load(model, state):
    state = strip_prefix(state)
    prefixes = ('temporal_transformer.', 'policy_head.')
    own = model.state_dict()
    needed = {k for k in own if k.startswith(prefixes)}
    supplied = {k for k in state if k.startswith(prefixes)}
    if not needed or supplied != needed:
        raise ValueError('Downstream keys mismatch: missing=%s unexpected=%s' %
                         (sorted(needed - supplied), sorted(supplied - needed)))
    if any(tuple(state[k].shape) != tuple(own[k].shape) for k in needed):
        raise ValueError('Downstream shape mismatch')
    merged = dict(own)
    merged.update({k: state[k] for k in needed})
    result = model.load_state_dict(merged, strict=True)
    return {'loaded_keys': len(needed), 'coverage': 1.0,
            'missing': list(result.missing_keys), 'unexpected': list(result.unexpected_keys),
            'spatial_keys_not_overwritten': len(own) - len(needed)}


def ifa_loss(last_feat, features, task_ids, current_task, info, negatives, gamma):
    """Mean over current samples and selected task pairs, retaining source angles.

    Language is modality 0 at time 0. Negative references retain old task 0.pt.
    """
    if features.ndim == 5:
        if features.shape[1] != 1:
            raise ValueError('Expected singleton saved feature batch axis')
        features = features.squeeze(1)
    if features.ndim != 4 or features.shape[2] != 4:
        raise ValueError('Expected [B,T,M=4,E]')
    ids = torch.as_tensor(task_ids, device=last_feat.device)
    selected = ids == current_task
    zero = last_feat.sum() * 0.0
    if info is None or not selected.any():
        return zero
    tids = [int(t) for t in info['task_ids']]
    if current_task not in tids:
        return zero
    pos = tids.index(current_task)
    mask = torch.as_tensor(info['selected_pair_mask'])
    peers = [t for i, t in enumerate(tids) if t != current_task and bool(mask[pos, i])]
    if not peers:
        return zero
    def norm(x):
        return x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    z = norm(last_feat[selected])
    positive = norm(features[selected, 0, 0, :].to(z).detach())
    theta_pos = torch.acos((z * positive).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
    values = []
    for peer in peers:
        negative = norm(negatives[peer].to(z).detach())
        theta_neg = torch.acos((z * negative).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        cos_ab = torch.as_tensor(info['sim_language'][pos, tids.index(peer)],
                                device=z.device, dtype=z.dtype)
        phi = torch.acos(cos_ab.clamp(-1 + 1e-6, 1 - 1e-6))
        values.append(torch.relu(theta_pos - theta_neg + gamma * phi).mean())
    return torch.stack(values).mean()


def metrics(matrix):
    r = np.asarray(matrix, dtype=float)
    if r.shape != (4, 4) or not np.isfinite(r[np.tril_indices(4)]).all():
        raise ValueError('Complete 4x4 lower triangle required')
    if not np.isnan(r[np.triu_indices(4, 1)]).all():
        raise ValueError('Unmeasured upper triangle must be NaN')
    if ((r[np.tril_indices(4)] < 0) | (r[np.tril_indices(4)] > 1)).any():
        raise ValueError('Rates must be in [0,1]')
    nbt = [float(np.mean(r[j, j] - r[j + 1:, j])) for j in range(3)]
    auc = [float(np.mean(r[j:, j])) for j in range(4)]
    return {'FWT': float(np.diag(r).mean()), 'NBT': float(np.mean(nbt)),
            'AUC': float(np.mean(auc)), 'FinalAvgSR': float(r[-1].mean()),
            'NBT_per_task': nbt, 'AUC_per_task': auc}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def transactional_buffer_update(feature_root, task, current_buffer, output_root, seed=42, rate=0.5, modal_contract=None):
    """Publish an immutable original-algorithm buffer generation with a marker.

    Repeated calls verify and return the committed generation without appending.
    A failed pending generation is preserved for inspection, never merged silently.
    The original buffer is never mutated. Not yet integrated into formal resume.
    """
    from libero.lifelong.algos.mlr import update_buffer_equal_quota_10, BUFFER_CAPACITY
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    generation = root / ('after_task_%d' % task)
    if generation.exists():
        manifest = json.loads((generation / 'manifest.json').read_text())
        if modal_contract is not None:
            from libero.lifelong.modal_contract import check_metadata
            check_metadata(manifest, modal_contract)
        if not manifest['complete'] or manifest['task'] != task:
            raise ValueError('Invalid buffer commit')
        for entry in manifest['files']:
            if sha256(generation / entry['name']) != entry['sha256']:
                raise ValueError('Committed buffer corrupted')
        return manifest
    pending = root / ('after_task_%d.pending' % task)
    pending.mkdir(exist_ok=False)
    for file in sorted(Path(current_buffer).glob('*.pt')):
        tid = int(file.name.split('__')[0][1:])
        if tid >= task:
            raise ValueError('Future/duplicate task in input buffer')
        os.link(str(file), str(pending / file.name))
    stats = update_buffer_equal_quota_10(str(feature_root), task, str(pending),
                                        capacity=BUFFER_CAPACITY, seed=seed, rate=rate)
    files = []
    for file in sorted(pending.glob('*.pt')):
        sample = torch.load(str(file), map_location='cpu')
        files.append({'name': file.name, 'sha256': sha256(file), 'bytes': file.stat().st_size,
                      'tensor_bytes': sum(x.numel() * x.element_size() for x in sample.values()
                                          if torch.is_tensor(x))})
    manifest = {'complete': True, 'task': task, 'seed': seed, 'storage_rate': rate,
                'capacity_files': BUFFER_CAPACITY, 'source_buffer': str(current_buffer),
                'generation': str(generation), 'original_algorithm_stats': stats, 'files': files,
                'disk_bytes': sum(v['bytes'] for v in files),
                'tensor_bytes': sum(v['tensor_bytes'] for v in files)}
    if modal_contract is not None:
        manifest.update(modal_contract=modal_contract, modal_fingerprint=modal_contract['fingerprint'])
    atomic_json(pending / 'manifest.json', manifest)
    os.replace(str(pending), str(generation))
    return json.loads((generation / 'manifest.json').read_text())
