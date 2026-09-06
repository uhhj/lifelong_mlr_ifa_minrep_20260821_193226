"""Frozen-checkpoint diagnostics; never constructs an optimizer or saves a model."""
import contextlib
import csv
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.algos.base import l2norm
from libero.lifelong.algos.mlr import build_ifa_similarity_matrix

OUT = ROOT / 'reports/corrected_baseline_diagnosis'
ARMS = ('mlr', 'release_ifa', 'corrected_ifa')


def run_dir(arm, seed):
    if arm == 'corrected_ifa':
        return ROOT / f'runs/own_reference_control/ifa/seed{seed}/run_001'
    return ROOT / f'runs/final_paper_text_faithful/{"mlr" if arm == "mlr" else "ifa"}/seed{seed}/run_001'


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probes(seed, cfg):
    meta = json.loads((ROOT / f'metadata/temporal_bins/object_seed{seed}.json').read_text())
    assert meta['seed'] == seed and meta['n_bins'] == 5
    assert cfg.data.frame_stack == 1 and cfg.data.seq_len == 8 and cfg.data.demo_limit == 10
    batches, manifest = {}, []
    for task in range(6, 10):
        path = ROOT / 'data/libero_object' / Path(meta['alignment'][str(task)]['dataset_path']).name
        lookup = defaultdict(list)
        with h5py.File(path, 'r') as file:
            demos = sorted(file['data'], key=lambda d: int(d.split('_')[-1]))[:10]
            for demo in demos:
                actions = np.asarray(file['data'][demo]['actions'], dtype=np.float32)
                for index in range(len(actions)):
                    seq = actions[np.minimum(np.arange(index, index + 8), len(actions) - 1)]
                    lookup[seq.tobytes()].append((demo, index, len(actions)))
        for bin_id in range(5):
            pool = sorted(int(k.split(':')[1]) for k, v in meta['records'].items()
                          if k.startswith(f'{task}:') and v == bin_id)
            selected = random.Random(20260906 + seed * 1000 + task * 10 + bin_id).sample(pool, 20)
            samples = []
            for position, record_id in enumerate(selected):
                feature_path = ROOT / f'features/object_seed{seed}/{task}/{record_id}.pt'
                data = torch.load(feature_path, map_location='cpu')
                matches = lookup[data['actions'].numpy().astype(np.float32).reshape(8, 7).tobytes()]
                assert len(matches) == 1, (feature_path, matches)
                demo, index, length = matches[0]
                progress = index / max(length - 1, 1)
                assert min(int(progress * 5), 4) == bin_id
                manifest.append(dict(seed=seed, task=task, bin=bin_id, batch=position // 10,
                                     feature=str(feature_path.relative_to(ROOT)), demo=demo,
                                     sequence_start=index, trajectory_length=length, progress=progress))
                samples.append(data)
            for repeat in range(2):
                batches[task, bin_id, repeat] = {
                    k: v.cuda() for k, v in default_collate(samples[repeat*10:(repeat+1)*10]).items()}
    return batches, manifest


def flat_grad(loss, params, retain=False):
    grad = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain)
    result = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                        for p, g in zip(params, grad)]).detach()
    assert torch.isfinite(loss) and torch.isfinite(result).all()
    return result


def relation(a, b):
    na, nb = float(a.norm()), float(b.norm())
    dot = float(torch.dot(a, b))
    return dict(dot=dot, cosine=dot/(na*nb) if min(na, nb) > 1e-10 else None,
                source_norm=na, target_norm=nb)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'COMPLETE.json').exists(), 'Do not overwrite a completed diagnosis'
    torch.set_num_threads(2)
    torch.manual_seed(20260906)
    bc_rows, pair_rows, ifa_rows, manifest, checks = [], [], [], [], []
    for seed in (42, 43, 44):
        cfg = EasyDict(json.loads((run_dir('corrected_ifa', seed) / 'config.json').read_text()))
        assert cfg.ifa_own_reference_fix and cfg.gamma == .3 and cfg.ifa_weight == .1
        batches, seed_manifest = probes(seed, cfg)
        manifest.extend(seed_manifest)
        write_csv('probe_manifest.csv', manifest)
        pairs = {}
        for current in (7, 8, 9):
            with contextlib.redirect_stdout(sys.stderr):
                info = build_ifa_similarity_matrix(current, cfg.feature_dir, cfg.buffer_dir)
            pairs[current] = []
            for a, b in info['selected_pairs']:
                if current not in (a, b):
                    continue
                old = b if a == current else a
                sim = float(info['sim_language'][info['task_ids'].index(current), info['task_ids'].index(old)])
                pairs[current].append((old, sim))
        expected = {7: [6], 8: [7] if seed == 42 else [], 9: []}
        assert {k: [x[0] for x in v] for k, v in pairs.items()} == expected
        print(f'[pairs] seed={seed} pairs={expected}', flush=True)
        for arm in ARMS:
            arm_cfg = EasyDict(json.loads((run_dir(arm, seed) / 'config.json').read_text()))
            with contextlib.redirect_stdout(sys.stderr):
                model = BriefPolicy(arm_cfg, arm_cfg.shape_meta).cuda().eval()
            assert not model.policy_head.low_eval_noise
            params = [p for p in model.parameters() if p.requires_grad]
            assert all(n.startswith(('temporal_transformer.', 'policy_head.'))
                       for n, p in model.named_parameters() if p.requires_grad)
            for stage in range(6, 10):
                path = run_dir(arm, seed) / f'mlrifa_{stage}_model_ep99.pth'
                ckpt = torch.load(path, map_location='cpu')
                state = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['state_dict'].items()}
                model.load_state_dict(state, strict=True)
                for bin_id, repeat in itertools.product(range(5), range(2)):
                    base = dict(arm=arm, seed=seed, checkpoint_stage=stage, epoch=99, bin=bin_id, batch=repeat)
                    grads, ifa_grads = {}, []
                    for task in range(6, 10):
                        data = batches[task, bin_id, repeat]
                        # BriefPolicy.forward only returns a distribution in train mode.
                        # Use its same components directly with dropout disabled.
                        latent = model.temporal_encode(data['feature'].squeeze(1))[:, -1, :]
                        distribution = model.policy_head(latent)
                        bc = model.policy_head.loss_fn(distribution, data['actions'].squeeze(1))
                        selected = pairs.get(task, [])
                        grads[task] = flat_grad(bc, params, retain=bool(selected))
                        bc_rows.append(dict(**base, task=task, bc=float(bc), gradient_norm=float(grads[task].norm())))
                        for old, sim in selected:
                            own = data['feature'].squeeze(1)[:, 0, 0, :].float()
                            neg = batches[old, bin_id, repeat]['feature'].squeeze(1)[:, 0, 0, :].float()
                            gn, rn, nn = (l2norm(l2norm(v)) for v in (latent, own, neg))
                            pos = torch.acos((gn*rn).sum(-1).clamp(-1+1e-6, 1-1e-6))
                            minus = torch.acos((gn*nn).sum(-1).clamp(-1+1e-6, 1-1e-6))
                            phi = torch.acos(torch.tensor(sim, device='cuda').clamp(-1+1e-6, 1-1e-6))
                            hinge = torch.relu(pos-minus+cfg.gamma*phi)
                            weighted = cfg.ifa_weight*hinge.mean()
                            g = flat_grad(weighted, params)
                            ifa_grads.append((task, old, g, float(weighted), float((hinge > 0).float().mean())))
                    for a, b in itertools.combinations(range(6, 10), 2):
                        pair_rows.append(dict(**base, task_a=a, task_b=b, **relation(grads[a], grads[b])))
                    for task, old, g, loss, active in ifa_grads:
                        for target in range(6, 10):
                            ifa_rows.append(dict(**base, current=task, selected_old=old, target=target,
                                                 pair_stage_matches_checkpoint=task == stage,
                                                 weighted_ifa=loss, hinge_active=active,
                                                 **relation(g, grads[target])))
                    del grads, ifa_grads
                assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
                checks.append(dict(arm=arm, seed=seed, stage=stage, strict_load=True, unchanged_tensors=True))
                print(f'[checkpoint] arm={arm} seed={seed} stage={stage} epoch=99 complete', flush=True)
                write_csv('bc_profiles.csv', bc_rows)
                write_csv('bc_gradient_pairs.csv', pair_rows)
                write_csv('ifa_gradient_pairs.csv', ifa_rows)
                del ckpt, state
            del model
            torch.cuda.empty_cache()
        del batches
    assert (len(manifest), len(checks), len(bc_rows), len(pair_rows), len(ifa_rows)) == (1200, 36, 1440, 2160, 1920)
    for rows in (bc_rows, pair_rows, ifa_rows):
        for seed in (42, 43, 44):
            groups = [[{k:v for k,v in r.items() if k != 'arm'} for r in rows
                       if r['seed'] == seed and r['checkpoint_stage'] == 6 and r['arm'] == arm] for arm in ARMS]
            assert groups[0] == groups[1] == groups[2], 'Task6 negative control differs'
    (OUT / 'COMPLETE.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',
        source='d75fb371cf575beb079ce064d70d2ae323da95dd', checkpoints=checks,
        probe_records=len(manifest), bc_rows=len(bc_rows), bc_pair_rows=len(pair_rows),
        ifa_pair_rows=len(ifa_rows), task6_exact_parity=True, optimizer_steps=0,
        mode='eval; full decoder/head gradients; descriptive offline probes'), indent=2)+'\n')


if __name__ == '__main__':
    main()
