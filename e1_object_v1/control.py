"""Single candidate correction. Native loss, AdamW and scheduling are callbacks."""
import copy
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    every: int = 10
    batch_size: int = 20
    tau: float = 1e-4
    tau_num: float = 1e-6
    rho: float = 0.25
    grad_floor: float = 1e-12


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])


@contextmanager
def deterministic_bc(model):
    """Keep training GMM semantics, disable only stochastic/stateful layers.

    Do not call model.eval(): BriefPolicy has a training-only forward contract.
    All mode flags, buffers, known action history, and RNG are restored exactly.
    """
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    history = {n: copy.deepcopy(getattr(model, n)) for n in ('latent_queue', 'history_buffer') if hasattr(model, n)}
    state = rng_state()
    try:
        model.training = True
        if hasattr(model, 'policy_head'): model.policy_head.training = True
        for m in model.modules():
            if isinstance(m, (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm)):
                m.training = False
        with torch.cuda.amp.autocast(enabled=False):
            yield
    finally:
        for m, flag in modes: m.training = flag
        with torch.no_grad():
            for b, old in buffers: b.copy_(old)
        for n, v in history.items(): setattr(model, n, v)
        restore_rng(state)


def norm32(values):
    return torch.sqrt(torch.stack([v.float().square().sum() for v in values]).sum())


def put(params, values):
    with torch.no_grad():
        for p, v in zip(params, values): p.copy_(v)


class Controller:
    def __init__(self, model, seed, settings=Settings()):
        self.model = model
        self.params = tuple(p for p in model.parameters() if p.requires_grad)
        assert self.params and all(p.dtype == torch.float32 for p in self.params)
        self.settings = settings
        self.sampling_rng = random.Random(1000003 + int(seed))

    def losses(self, data, grad=False):
        with deterministic_bc(self.model), torch.set_grad_enabled(grad):
            values, _ = self.model.compute_loss(data, None, None, reduction='none')
            if values.ndim != 1: raise ValueError('E1 requires one original GMM NLL per window')
            if not grad: return values.detach()
            # autograd.grad never accumulates into native .grad.
            g = torch.autograd.grad(values.mean(), self.params, allow_unused=True)
            return values.detach(), tuple(torch.zeros_like(p) if x is None else x.detach() for p, x in zip(self.params, g))

    def update(self, native, count, sample, metadata):
        """sample uses only this controller's local RNG and legal old buffer."""
        s = self.settings
        before_count = count()
        if not s.enabled or (before_count + 1) % s.every:
            return native(), None
        # Sampling/collation must not consume the native training RNG stream.
        pre_begin = time.monotonic()
        state = rng_state()
        try: data, ids = sample(self.sampling_rng, s.batch_size)
        finally: restore_rng(state)
        if data is None: return native(), None
        begin = pre_begin
        theta0 = tuple(p.detach().clone() for p in self.params)
        l0s = self.losses(data)
        overhead_pre = time.monotonic() - begin
        result = native()                 # exactly one original MLR+IFA step
        if count() != before_count + 1:
            if count() == before_count: return result, None
            raise RuntimeError('E1_NATIVE_OPTIMIZER_COUNT_MISMATCH')
        begin = time.monotonic()
        candidate = tuple(p.detach().clone() for p in self.params)
        if not all(bool(torch.isfinite(p).all()) for p in candidate):
            raise FloatingPointError('Nonfinite native model; not an ordinary E1 skip')
        delta_norm = float(norm32([c - p for c, p in zip(candidate, theta0)]))
        row = dict(metadata, update=count(), sample_ids=ids, L0=None, Lc=None, Lf=None,
                   q=None, epsilon=None, xi=None, excess=None, grad_norm=None,
                   delta_norm=delta_norm, correction_norm=0.0, actual_correction_norm=0.0,
                   triggered=False, capped=False, accepted=False, residual_excess=None, reason='')
        try:
            if not bool(torch.isfinite(l0s).all()):
                row['reason'] = 'NONFINITE_CHECK_LOSS'; return result, row
            l0 = float(l0s.mean()); scale = max(1., float(l0s.abs().mean()))
            lc = float(self.losses(data).mean())
            row.update(L0=l0, Lc=lc if math.isfinite(lc) else None, epsilon=s.tau*scale, xi=s.tau_num*scale)
            if not math.isfinite(lc):
                row['reason'] = 'NONFINITE_CHECK_LOSS'; return result, row
            q = lc-l0; v = q-row['epsilon']; row.update(q=q, excess=max(0.,v))
            if v <= 0:
                row['reason'] = 'WITHIN_TOLERANCE'; return result, row
            row['triggered'] = True
            if delta_norm == 0 or not math.isfinite(delta_norm) or s.rho == 0:
                row['reason'] = 'ZERO_UPDATE_OR_ZERO_CORRECTION_BUDGET'; return result, row
            _, g = self.losses(data, grad=True)
            gn = float(norm32(g))
            if not math.isfinite(gn) or gn < s.grad_floor or gn == 0:
                row.update(grad_norm=gn if math.isfinite(gn) else None, reason='INVALID_CHECK_GRADIENT'); return result, row
            length = min(v/gn, s.rho*delta_norm)
            correction = tuple(-length*(a/gn) for a in g)
            if not all(bool(torch.isfinite(a).all()) for a in correction):
                row['reason'] = 'NONFINITE_CORRECTION'; return result, row
            row.update(grad_norm=gn, capped=v/gn > s.rho*delta_norm,
                       correction_norm=float(norm32(correction)))
            put(self.params, [c+r for c,r in zip(candidate, correction)])
            row['actual_correction_norm'] = float(norm32([p-c for p,c in zip(self.params,candidate)]))
            lf = float(self.losses(data).mean())
            row['Lf'] = lf if math.isfinite(lf) else None
            row['accepted'] = math.isfinite(lf) and lf <= lc-row['xi']
            if row['accepted']:
                row.update(reason='ACCEPTED', residual_excess=max(0.,lf-l0-row['epsilon']))
            else:
                put(self.params, candidate)  # exact copy, never add then subtract
                row.update(reason='RESTORED_NATIVE_CANDIDATE', residual_excess=max(0.,v))
            return result, row
        except Exception:
            put(self.params, candidate)
            raise
        finally:
            row['overhead_seconds'] = overhead_pre + time.monotonic()-begin
