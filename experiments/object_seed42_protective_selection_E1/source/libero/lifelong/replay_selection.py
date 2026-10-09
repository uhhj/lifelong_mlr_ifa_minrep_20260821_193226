"""E1 same-task replay replacement. No optimizer, buffer update or rollout API.

The caller supplies an immutable, stage-legal catalog and its verified payload
loader. This module never discovers files or reads discarded old features.
"""
import copy
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


@dataclass(frozen=True)
class SelectionConfig:
    enabled: bool = True
    refresh_interval_steps: int = 500
    refresh_at_stage_start: bool = True
    reference_per_old_task: int = 16
    candidates_per_old_task: int = 64
    current_reference_size: int = 16
    priority_replace_probability: float = 0.5
    conflict_coefficient: float = 1.0
    softmax_temperature: float = 1.0
    numerical_epsilon: float = 1e-8
    gradient_scope: str = 'policy_head'
    preserve_task_slots: bool = True
    fallback: str = 'original_retrieval'

    def __post_init__(self):
        assert self.refresh_interval_steps > 0
        assert min(self.reference_per_old_task, self.candidates_per_old_task,
                   self.current_reference_size) > 0
        assert 0 <= self.priority_replace_probability <= 1
        assert self.softmax_temperature > 0 and self.numerical_epsilon > 0
        assert self.conflict_coefficient >= 0 and self.gradient_scope == 'policy_head'
        assert self.preserve_task_slots and self.fallback == 'original_retrieval'


@contextmanager
def isolated_scoring(policy):
    """Restore *every* module mode and all model RNG streams, also on errors."""
    modes = [(m, m.training) for m in policy.modules()]
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        policy.eval()
        # Full training density, including when low_eval_noise=True.
        policy.policy_head.train(True)
        with torch.enable_grad():
            yield
    finally:
        for m, flag in modes:
            m.training = flag
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def head_gradient(policy, payloads):
    """Exact baseline BC target slice; temporal decoder has no autograd graph."""
    params = tuple(policy.policy_head.parameters())
    if not params or not all(p.requires_grad for p in params):
        raise RuntimeError('Scoring requires the baseline trainable policy head')
    data = _to_device(default_collate(payloads), params[0].device)
    if not torch.isfinite(data['feature']).all().item() or not torch.isfinite(data['actions']).all().item():
        return None
    # Match BasePolicy.compute_loss's squeeze(1), without BriefPolicy.forward.
    with torch.no_grad():
        features = policy.temporal_encode(data['feature'].squeeze(1))[:, -1, :].detach()
    if not torch.isfinite(features).all().item():
        return None
    density = policy.policy_head(features, train=True)
    loss = policy.policy_head.loss_fn(density, data['actions'].squeeze(1), reduction='mean')
    if not torch.isfinite(loss).item():
        return None
    grads = torch.autograd.grad(loss, params, allow_unused=False)
    vector = torch.cat([g.detach().reshape(-1) for g in grads])
    return vector if torch.isfinite(vector).all().item() else None


def unit(vector, epsilon):
    if vector is None or not torch.isfinite(vector).all().item():
        return None
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm).item() or norm.item() <= epsilon:
        return None
    return vector / norm


def protective_score(candidate, historical, current, alpha=1.0, epsilon=1e-8):
    c, h = unit(candidate, epsilon), unit(historical, epsilon)
    n = unit(current, epsilon)
    if c is None or h is None:
        return None
    score = torch.dot(c, h)
    if n is not None:
        score = score - alpha * torch.clamp(-torch.dot(c, n), min=0)
    value = float(score.item())
    return value if math.isfinite(value) else None


def language_reference(feature):
    # A single payload has [1,T,M,E]; batch indexing removes only B.
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature.squeeze(0)
    if feature.ndim != 3:
        raise ValueError('Expected one [1,T,M,E] or [T,M,E] latent payload')
    return feature[0, 0, :]


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, (list, tuple)) else value


class ProtectiveReplaySelector:
    def __init__(self, policy, seed, current_task, old_catalog, current_ids,
                 payload_loader, catalog_fingerprint, config=None):
        self.policy = getattr(policy, 'module', policy)
        self.config = config or SelectionConfig()
        self.current_task = int(current_task)
        self.old = {int(t): tuple(ids) for t, ids in sorted(old_catalog.items()) if ids}
        assert all(t < current_task for t in self.old), 'Future-task leakage'
        flat = [i for ids in self.old.values() for i in ids]
        self.current = tuple(current_ids)
        assert len(flat) == len(set(flat)) and len(self.current) == len(set(self.current))
        assert set(flat).isdisjoint(self.current)
        self.load = payload_loader
        self.catalog_fingerprint = catalog_fingerprint
        self.rng = random.Random(int(seed))  # Independent; never random.seed().
        self.last_refresh_step = None
        self.distributions = {}
        self.refresh_record = None
        self.totals = dict(refreshes=0, old_slots=0, priority_draws=0,
                           assigned_slots=0, first_reference_veto=0,
                           invalid_candidates=0, fallback_refreshes=0,
                           new_direction_fallbacks=0, io_seconds=0.,
                           scoring_seconds=0., replacement_seconds=0.)

    def should_refresh(self, stage_optimizer_step):
        if not self.config.enabled or not self.old or self.config.priority_replace_probability == 0:
            return False
        if self.last_refresh_step is None:
            return self.config.refresh_at_stage_start or stage_optimizer_step >= self.config.refresh_interval_steps
        return stage_optimizer_step - self.last_refresh_step >= self.config.refresh_interval_steps

    def _load_many(self, ids):
        begin = time.perf_counter()
        result = [self.load(i) for i in ids]
        self.totals['io_seconds'] += time.perf_counter() - begin
        return result

    def refresh(self, stage_optimizer_step):
        if not self.should_refresh(stage_optimizer_step):
            return None
        begin = time.perf_counter()
        cfg = self.config
        record = dict(step=int(stage_optimizer_step), references={}, candidates={},
                      scores={}, probabilities={}, valid_ids={}, invalid_candidates={}, reason='OK')
        refs, candidates = {}, {}
        for task, ids in self.old.items():
            n_ref = min(cfg.reference_per_old_task, max(1, len(ids)-1))
            sampled = self.rng.sample(ids, min(len(ids), n_ref + cfg.candidates_per_old_task))
            refs[task], candidates[task] = sampled[:n_ref], sampled[n_ref:]
            record['references'][str(task)] = refs[task]
            record['candidates'][str(task)] = candidates[task]
        new_ids = self.rng.sample(self.current, min(len(self.current), cfg.current_reference_size))
        record['current_reference'] = new_ids
        distributions = {}
        with isolated_scoring(self.policy):
            h = None
            for task, ids in refs.items():
                g = head_gradient(self.policy, self._load_many(ids))
                if g is None:
                    h = None
                    break
                h = g / len(refs) if h is None else h + g / len(refs)
            if unit(h, cfg.numerical_epsilon) is None:
                record['reason'] = 'INVALID_HISTORICAL_DIRECTION'
                self.totals['fallback_refreshes'] += 1
            else:
                n = head_gradient(self.policy, self._load_many(new_ids)) if new_ids else None
                if unit(n, cfg.numerical_epsilon) is None:
                    n = None
                    self.totals['new_direction_fallbacks'] += 1
                    record['reason'] = 'HISTORY_ONLY_INVALID_CURRENT_DIRECTION'
                for task, ids in candidates.items():
                    valid_ids, scores, invalid = [], [], []
                    for item_id in ids:
                        g = head_gradient(self.policy, self._load_many([item_id]))
                        score = protective_score(g, h, n, cfg.conflict_coefficient, cfg.numerical_epsilon)
                        if score is None:
                            invalid.append(item_id)
                        else:
                            valid_ids.append(item_id)
                            scores.append(score)
                        del g
                    self.totals['invalid_candidates'] += len(invalid)
                    record['invalid_candidates'][str(task)] = invalid
                    if scores:
                        scaled = [s / cfg.softmax_temperature for s in scores]
                        peak = max(scaled)
                        weights = [math.exp(s-peak) for s in scaled]
                        total = sum(weights)
                        probs = [x/total for x in weights]
                        distributions[task] = dict(ids=valid_ids, scores=scores, probabilities=probs)
                        record['valid_ids'][str(task)] = valid_ids
                        record['scores'][str(task)] = scores
                        record['probabilities'][str(task)] = probs
        self.distributions = distributions
        self.last_refresh_step = int(stage_optimizer_step)
        self.totals['refreshes'] += 1
        # CUDA tensor scalar checks above synchronize all scoring kernels.
        self.totals['scoring_seconds'] += time.perf_counter()-begin
        self.refresh_record = record
        return copy.deepcopy(record)

    def replace_old_slots(self, batch):
        if not self.config.enabled or self.config.priority_replace_probability == 0:
            return batch, []
        begin = time.perf_counter()
        payload, tasks = batch
        output = None
        events = []
        for slot, value in enumerate(tasks):
            task = int(value)
            if task >= self.current_task:
                assert task == self.current_task, 'Future task in original batch'
                continue
            self.totals['old_slots'] += 1
            distribution = self.distributions.get(task)
            if not distribution or self.rng.random() >= self.config.priority_replace_probability:
                continue
            item_id = self.rng.choices(distribution['ids'], weights=distribution['probabilities'], k=1)[0]
            self.totals['priority_draws'] += 1
            sample = self._load_many([item_id])[0]
            assert set(sample) == set(payload), 'Payload schema changed'
            if slot == 0 and not torch.equal(language_reference(payload['feature'][0]).cpu(), language_reference(sample['feature']).cpu()):
                self.totals['first_reference_veto'] += 1
                events.append(dict(slot=slot, task=task, candidate=item_id, applied=False, reason='IFA_LANGUAGE_REFERENCE'))
                continue
            if output is None:
                output = {k:v.clone() for k,v in payload.items()}
            for key, value in sample.items():
                assert torch.is_tensor(value) and value.shape == output[key][slot].shape
                assert value.dtype == output[key].dtype, 'Payload dtype changed'
                output[key][slot].copy_(value.to(output[key].device))
            self.totals['assigned_slots'] += 1
            events.append(dict(slot=slot, task=task, candidate=item_id, applied=True))
        self.totals['replacement_seconds'] += time.perf_counter()-begin
        return ([output, tasks] if output is not None else batch), events

    def state_dict(self):
        return copy.deepcopy(dict(version=1, config=asdict(self.config), current_task=self.current_task,
            catalog_fingerprint=self.catalog_fingerprint, old_catalog=self.old, current_ids=self.current,
            rng=self.rng.getstate(), last_refresh_step=self.last_refresh_step,
            distributions=self.distributions, refresh_record=self.refresh_record, totals=self.totals))

    def load_state_dict(self, state):
        assert state['version'] == 1 and state['config'] == asdict(self.config)
        assert state['current_task'] == self.current_task
        assert state['catalog_fingerprint'] == self.catalog_fingerprint
        assert {int(k):tuple(v) for k,v in state['old_catalog'].items()} == self.old
        assert tuple(state['current_ids']) == self.current
        distributions = {int(k):v for k,v in state['distributions'].items()}
        for task, d in distributions.items():
            assert set(d['ids']) <= set(self.old[task])
            assert len(d['ids']) == len(d['probabilities']) == len(d['scores'])
            assert all(math.isfinite(p) and p > 0 for p in d['probabilities'])
            assert abs(sum(d['probabilities'])-1) < 1e-10
        self.rng.setstate(_tuple_tree(state['rng']))
        self.last_refresh_step = state['last_refresh_step']
        self.distributions = copy.deepcopy(distributions)
        self.refresh_record = copy.deepcopy(state['refresh_record'])
        self.totals = copy.deepcopy(state['totals'])
