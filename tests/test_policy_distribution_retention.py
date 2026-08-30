import math

import pytest
import torch

from libero.lifelong.analysis.policy_distribution_retention import (
    PRIMARY_SIGNALS,
    aggregate_phase_metric,
    decide_h19,
    deterministic_gmm_forward_kl,
    diagonal_gaussian_forward_kl,
    diagonal_gaussian_symmetric_kl,
    diagonal_gaussian_w2_sq,
    matched_signal_contrasts,
    quantile,
    rank_primary_signals,
    select_rollout_normal,
)
from libero.lifelong.models.policy_head import GMMHead


def stats(means, stds, logits):
    mu, std, mode = select_rollout_normal(means, stds, logits)
    return {
        "means": means,
        "stds": stds,
        "logits": logits,
        "rollout_mu": mu,
        "rollout_std": std,
        "rollout_mode": mode,
    }


def test_gaussian_kl_identity():
    mu = torch.tensor([[0.2, -0.1]])
    std = torch.tensor([[0.7, 1.4]])
    assert diagonal_gaussian_forward_kl(mu, std, mu, std).item() == pytest.approx(0.0)
    assert diagonal_gaussian_symmetric_kl(mu, std, mu, std).item() == pytest.approx(0.0)


def test_gaussian_known_shift():
    zero = torch.zeros(1, 1)
    one = torch.ones(1, 1)
    assert diagonal_gaussian_forward_kl(zero, one, one, one).item() == pytest.approx(0.5)


def test_w2_identity_and_known_shift():
    zero = torch.zeros(1, 2)
    one = torch.ones(1, 2)
    assert diagonal_gaussian_w2_sq(zero, one, zero, one).item() == 0.0
    assert diagonal_gaussian_w2_sq(zero, one, one, one).item() == pytest.approx(2.0)


def test_selected_rollout_normal_matches_gmm_head_eval_distribution():
    torch.manual_seed(3)
    head = GMMHead(input_size=4, output_size=7, hidden_size=8, num_layers=1, num_modes=3)
    head.eval()
    inputs = torch.randn(5, 4)
    means, stds, logits = head.forward_fn(inputs)
    mu, std, mode = select_rollout_normal(means, stds, logits)
    distribution = head(inputs, train=False)
    assert torch.equal(mode, logits.argmax(-1))
    assert torch.allclose(mu, distribution.base_dist.loc)
    assert torch.allclose(std, distribution.base_dist.scale)


def test_deterministic_gmm_mc_repeatability():
    means = torch.tensor([[[0.0], [1.0]], [[0.5], [-0.5]]])
    stds = torch.ones_like(means)
    logits = torch.zeros(2, 2)
    anchor = stats(means, stds, logits)
    current = stats(means + 0.25, stds, logits)
    first = deterministic_gmm_forward_kl(anchor, current, 64, [11, 12])
    second = deterministic_gmm_forward_kl(anchor, current, 64, [11, 12])
    assert torch.equal(first, second)


def test_full_gmm_identity_near_zero_and_shift_larger():
    means = torch.tensor([[[0.0], [1.0]]])
    stds = torch.ones_like(means)
    logits = torch.zeros(1, 2)
    anchor = stats(means, stds, logits)
    identity = deterministic_gmm_forward_kl(anchor, anchor, 64, 9)
    shifted = deterministic_gmm_forward_kl(
        anchor, stats(means + 2.0, stds, logits), 64, 9
    )
    assert abs(identity.mean().item()) <= 5e-3
    assert shifted.mean().item() > identity.mean().item()


def test_phase_aggregation_mean_worst_late():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bins = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    result = aggregate_phase_metric(values, bins)
    assert result["mean"] == pytest.approx(5.5)
    assert result["worst_bin"] == pytest.approx(9.5)
    assert result["late_bin"] == pytest.approx(9.5)


def test_q90_linear_semantics():
    assert quantile(range(10), 0.9) == pytest.approx(8.1)


def test_matched_contrast_sign_and_dangerous_bw():
    signal = [
        {"method": "temporal_v1", "seed": 42, "stage": 7, "epoch": 70, "task": 6, "x": 2.0},
        {"method": "zero_guard", "seed": 42, "stage": 7, "epoch": 70, "task": 6, "x": 1.0},
    ]
    rollout = [
        {"method": "temporal_v1", "seed": 42, "stage": 7, "epoch": 70, "task": 6, "success_pct": 60},
        {"method": "zero_guard", "seed": 42, "stage": 7, "epoch": 70, "task": 6, "success_pct": 50},
    ]
    rows = matched_signal_contrasts(signal, rollout, "x", "toy")
    assert rows[0]["quadrant"] == "BW"
    assert rows[0]["dangerous_mismatch"] == 1


def make_summary(auc, rho, concordance, dangerous, delta=0.2):
    return {
        "roc_auc": auc, "spearman": rho, "sign_concordance": concordance,
        "dangerous_rate": dangerous, "delta_auc": delta,
    }


def test_h19_supported_not_supported_and_inconclusive():
    from libero.lifelong.analysis.policy_distribution_retention import decide_h19
    supported = []
    seeds = []
    for index, name in enumerate(PRIMARY_SIGNALS):
        row = make_summary(0.75, 0.35, 0.68, 0.2) if index == 0 else make_summary(0.62, 0.22, 0.6, 0.3)
        supported.append({"signal": name, **row})
        for seed in (42, 43, 44):
            seeds.append({"signal": name, "seed": seed, "roc_auc": row["roc_auc"], "sign_concordance": row["sign_concordance"]})
    assert decide_h19(supported, {}, seeds, True).endswith("_SUPPORTED")
    weak = [{"signal": name, **make_summary(0.55, 0.1, 0.55, 0.4, 0.0)} for name in PRIMARY_SIGNALS]
    assert decide_h19(weak, {}, seeds, True).endswith("_NOT_SUPPORTED")
    mixed = [{"signal": name, **make_summary(0.65, 0.25, 0.6, 0.3, 0.1)} for name in PRIMARY_SIGNALS]
    assert decide_h19(mixed, {}, seeds, True).endswith("_INCONCLUSIVE")


def test_primary_ranking_tie_breaks_by_declared_order():
    rows = [
        {"signal": name, **make_summary(0.7, 0.3, 0.65, 0.2)}
        for name in reversed(PRIMARY_SIGNALS)
    ]
    assert rank_primary_signals(rows)[0]["signal"] == PRIMARY_SIGNALS[0]
