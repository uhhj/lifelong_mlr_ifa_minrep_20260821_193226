"""ZR-v2 protocol reference; NOT an integrated LIBERO training/evaluation runner.

The supplied functions use real values only when a caller supplies them. Tests use
synthetic arrays. Gate construction is float64; residual matrices are float32.
"""
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


def normalize_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 2-D instruction matrix")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero instruction vector")
    return a / norms


def extend_gates(old_raw: np.ndarray, current_raw: np.ndarray,
                 old_gates: np.ndarray, base_count: int = 6,
                 min_novelty: float = 1e-2, max_gain: float = 100.0
                 ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Use only past+current canonical instructions. No future vectors or labels.

    old_gates has one row per previous incremental instruction, in arrival order.
    Raises on an inadmissible geometry; never snaps small responses to zero.
    """
    old = normalize_rows(old_raw)
    cur = normalize_rows(np.asarray(current_raw, dtype=np.float64).reshape(1, -1))[0]
    if old.shape[1] != cur.size or len(old) < base_count:
        raise ValueError("Instruction dimensions/count mismatch")
    w = np.asarray(old_gates, dtype=np.float64)
    if w.shape != (len(old) - base_count, cur.size) or not np.isfinite(w).all():
        raise ValueError("Old gate registry does not match arrived instructions")
    _, singular, vh = np.linalg.svd(old, full_matrices=False)
    threshold = max(1e-12, 1e-10 * float(singular[0]))
    if np.any((singular >= threshold / 10) & (singular <= threshold * 10)):
        raise ValueError("Numerical rank is ambiguous under the frozen criterion")
    basis = vh[singular > threshold]
    v = cur - basis.T @ (basis @ cur)
    rho = float(np.linalg.norm(v))
    if rho < min_novelty:
        raise ValueError("New instruction is too close to the protected span")
    denom = float(v @ cur)
    if denom <= 0:
        raise ValueError("Invalid projection denominator")
    new_gate = v / denom
    # Use pre-update rows; each row changes analytically, not via a learned router.
    updated_old = w - (w @ cur)[:, None] * new_gate[None, :]
    new_w = np.vstack([updated_old, new_gate])
    gains = np.linalg.norm(new_w, axis=1)
    if np.any(gains > max_gain):
        raise ValueError("Updated gate gain exceeds the frozen limit")
    all_e = np.vstack([old, cur])
    wanted = np.eye(len(all_e), dtype=np.float64)[:, base_count:]
    responses = all_e @ new_w.T
    err = float(np.max(np.abs(responses - wanted)))
    if err > 1e-10:
        raise ValueError("Canonical bidirectional response invariant failed")
    return new_w, dict(novelty=rho, rank=int(len(basis)), singular_values=singular.tolist(),
                       rank_threshold=threshold, gate_gains=gains.tolist(),
                       canonical_error=err, responses=responses.tolist())


class LowRankMean(nn.Module):
    def __init__(self, task_id: int, experiment_seed: int = 43, rank: int = 8):
        super().__init__()
        gen = torch.Generator(device='cpu').manual_seed(experiment_seed + task_id)
        self.V = nn.Parameter(torch.randn(rank, 512, generator=gen) * 0.02)
        self.U = nn.Parameter(torch.zeros(7, rank))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return (q @ self.V.T) @ self.U.T


class MeanCalibrationBank(nn.Module):
    """Finite-stage bank. Assemble/reload a stage before constructing AdamW.

    Task IDs are used ONLY to restrict training gradient ownership. Forward
    values and inference gates always depend on the actual language tensor.
    """
    def __init__(self, task_ids: Sequence[int], gates: np.ndarray, tau: float = 0.1,
                 experiment_seed: int = 43):
        super().__init__()
        if not task_ids or len(set(task_ids)) != len(task_ids) or not (tau > 0):
            raise ValueError("Invalid task registry or correction bound")
        a = np.asarray(gates, dtype=np.float64)
        if a.shape != (len(task_ids), 512) or not np.isfinite(a).all():
            raise ValueError("Gate dimensions mismatch")
        self.task_ids = tuple(int(t) for t in task_ids)
        self.tau = float(tau)
        self.register_buffer('gate_w', torch.from_numpy(a.copy()))
        self.branches = nn.ModuleDict({str(t): LowRankMean(t, experiment_seed) for t in task_ids})

    def forward(self, q: torch.Tensor, e: torch.Tensor,
                training_task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gate_w.dtype != torch.float64:
            raise RuntimeError("FP64 gates were downcast")
        if q.ndim != 2 or q.shape[1] != 512 or e.shape != q.shape:
            raise ValueError("Expected matching [B,512] frozen features and language")
        if not torch.isfinite(q).all() or not torch.isfinite(e).all():
            raise ValueError("Nonfinite features")
        if training_task_ids is not None:
            if training_task_ids.shape != (len(q),):
                raise ValueError("Training task labels have the wrong shape")
            observed = set(int(t) for t in training_task_ids.detach().cpu().tolist())
            if not observed.issubset(set(self.task_ids)):
                raise ValueError("Unknown calibration task in this stage")
        ee = e.detach().double()
        norm = ee.norm(dim=-1, keepdim=True)
        if bool((norm <= 1e-12).any()):
            raise ValueError("Zero language vector")
        response = (ee / norm) @ self.gate_w.T
        total = torch.zeros(len(q), 7, dtype=torch.float64, device=q.device)
        for col, t in enumerate(self.task_ids):
            value = self.branches[str(t)](q.detach())
            if training_task_ids is not None:
                own = training_task_ids == t
                # Explicit absent-branch detach -> grad=None, not zero-gradient
                # AdamW decay. The forward value is unchanged in both cases.
                value = torch.where(own[:, None], value, value.detach()) if bool(own.any()) else value.detach()
            total = total + response[:, col, None] * value.double()
        return (self.tau * torch.tanh(total)).to(q.dtype)


def corrected_gmm(mean_preactivation: torch.Tensor, scales: torch.Tensor,
                  logits: torch.Tensor, delta: torch.Tensor) -> Dict[str, object]:
    """scales/logits must be actual frozen public-head outputs, not re-estimates."""
    b = len(delta)
    if mean_preactivation.shape != (b, 5, 7) or scales.shape != (b, 5, 7):
        raise ValueError("Expected five 7-D components")
    if logits.shape != (b, 5) or delta.shape != (b, 7):
        raise ValueError("Output shapes do not match")
    if not all(bool(torch.isfinite(x).all()) for x in (mean_preactivation, scales, logits, delta)):
        raise ValueError("Nonfinite distribution inputs")
    if bool((scales <= 0).any()):
        raise ValueError("Invalid public scales")
    means = (mean_preactivation + delta[:, None, :]).tanh()
    dist = MixtureSameFamily(Categorical(logits=logits), Independent(Normal(means, scales), 1))
    return dict(means=means, scales=scales, logits=logits,
                probs=logits.softmax(-1), selected=logits.argmax(-1), gmm=dist)
