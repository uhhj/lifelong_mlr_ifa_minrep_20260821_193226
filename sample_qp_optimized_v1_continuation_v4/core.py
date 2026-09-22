"""Device-resident parameter operations; original double-precision QP algebra.

No gradient caching, reference subsampling, altered solver tolerance or optimizer
replacement. Historical implementations remain the numerical reference.
"""
import contextlib
import time
import numpy as np
import torch
from .solver import solve


class Timings:
    """Optional synchronized phase profiling, separate from speed benchmarks."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.seconds = {}

    @contextlib.contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter()-start


class Workspace:
    def __init__(self, parameters, max_reference_bytes=8*2**30):
        self.parameters = tuple(parameters)
        assert self.parameters and all(p.requires_grad for p in self.parameters)
        self.device = self.parameters[0].device
        assert all(p.device == self.device and p.dtype == torch.float32 for p in self.parameters)
        self.size = sum(p.numel() for p in self.parameters)
        self.max_reference_bytes = max_reference_bytes
        self.H = None

    def snapshot_parameters(self):
        # Only trainable parameters are needed for an actual AdamW displacement.
        return tuple(p.detach().clone() for p in self.parameters)

    def displacement(self, before):
        assert len(before) == len(self.parameters)
        return torch.cat([(p.detach()-old).reshape(-1)
                          for p, old in zip(self.parameters, before)])

    def subtract_correction(self, correction):
        assert correction.numel() == self.size
        assert correction.device == self.device and correction.dtype == torch.float32
        offset = 0
        with torch.no_grad():
            for p in self.parameters:
                c = correction[offset:offset+p.numel()].view_as(p)
                # Match legacy native_parameter + (-correction), without CPU copies.
                p.copy_(p.detach() + (-c))
                offset += p.numel()

    def sample_gradients(self, batches, loss_fn, count):
        assert count > 0 and self.size*count*4 <= self.max_reference_bytes
        if self.H is None or self.H.shape != (self.size, count):
            self.H = torch.empty((self.size, count), device=self.device, dtype=torch.float32)
        column = 0
        for batch in batches:
            losses = loss_fn(batch)
            assert losses.ndim == 1 and len(losses) == 8
            assert bool(torch.isfinite(losses).all())
            for i in range(len(losses)):
                gradients = torch.autograd.grad(losses[i], self.parameters,
                    retain_graph=i < len(losses)-1, allow_unused=True)
                flat = torch.cat([(torch.zeros_like(p) if z is None else z).reshape(-1)
                                  for z, p in zip(gradients, self.parameters)])
                self.H[:, column].copy_(flat)
                column += 1
                del gradients, flat
            del losses
        assert column == count
        return self.H


def inner(H, u=None, gram=False):
    # Preserve legacy chunk boundaries, precision, reduction order and GEMM shapes.
    K = torch.zeros((H.shape[1], H.shape[1]), device=H.device, dtype=torch.float64) if gram else None
    f = torch.zeros(H.shape[1], device=H.device, dtype=torch.float64) if u is not None else None
    for start in range(0, H.shape[0], 65536):
        B = H[start:start+65536].double()
        if gram:
            K.add_(B.T @ B)
        if u is not None:
            f.add_(B.T @ u[start:start+65536].double())
    return (None if K is None else K.cpu().numpy(),
            None if f is None else f.cpu().numpy())


def project(H, K, displacement, timings=None, diag_dir=None, context=None):
    t = timings or Timings()
    with t.phase('projection_inner'):
        _, f = inner(H, displacement)
        un = float(displacement.double().norm())
    with t.phase('qp_solver'):
        sol = solve(K, f, un, diag_dir=diag_dir, context=context)  # Same SLSQP + polishing and frozen KKT tolerance.
    with t.phase('correction_and_residual'):
        lam = torch.tensor(sol['lam'], device=H.device, dtype=torch.float64)
        correction = torch.empty_like(displacement)
        for start in range(0, len(displacement), 65536):
            correction[start:start+65536] = (H[start:start+65536].double() @ lam).float()
        _, written = inner(H, displacement-correction)
        hn = np.sqrt(np.maximum(np.diag(K), 0))
        valid = np.asarray(sol['valid'], dtype=int)
        den = np.maximum(hn*max(un, 1e-30), 1e-30)
        residual = float(np.maximum(written[valid]/den[valid], 0).max()) if len(valid) else 0.
        assert residual <= 5e-6, ('Original float32 projection tolerance failed', residual)
    sol.update(f=f.tolist(), u_norm=un, written_normalized_rate_max=residual)
    return correction, sol
