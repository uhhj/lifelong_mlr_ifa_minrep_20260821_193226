import math
import statistics
from collections import Counter, defaultdict

import torch

from libero.lifelong.analysis.bc_rollout_retention_alignment import (
    pearson,
    quadrant,
    spearman,
)
from libero.lifelong.analysis.retention_score_calibration import (
    average_precision,
    roc_auc,
)


PRIMARY_SIGNALS = (
    "rollout_kl_global_mean",
    "rollout_kl_global_q90",
    "rollout_kl_phase_worst_bin",
    "rollout_kl_phase_late_bin",
    "rollout_w2_global_mean",
    "rollout_w2_global_q90",
    "rollout_w2_phase_worst_bin",
    "rollout_w2_phase_late_bin",
)
BC_CONTROLS = (
    "bc_global_mean",
    "bc_global_q90",
    "bc_phase_mean",
    "bc_phase_worst_bin",
    "bc_phase_late_bin",
)


@torch.no_grad()
def extract_policy_head_stats(policy, features):
    temporal = policy.temporal_encode(features)
    last_feat = temporal[:, -1, :]
    means, stds, logits = policy.policy_head.forward_fn(last_feat)
    mu, std, mode = select_rollout_normal(means, stds, logits)
    return {
        "means": means.detach(),
        "stds": stds.detach(),
        "logits": logits.detach(),
        "rollout_mu": mu.detach(),
        "rollout_std": std.detach(),
        "rollout_mode": mode.detach(),
    }


def select_rollout_normal(means, stds, logits):
    idx = logits.argmax(dim=-1)
    batch = torch.arange(means.shape[0], device=means.device)
    return means[batch, idx], stds[batch, idx], idx


def diagonal_gaussian_forward_kl(mu_p, std_p, mu_q, std_q):
    mu_p = mu_p.double()
    mu_q = mu_q.double()
    std_p = std_p.double()
    std_q = std_q.double()
    var_p = std_p.square()
    var_q = std_q.square()
    diff2 = (mu_p - mu_q).square()
    result = (
        torch.log(std_q / std_p)
        + (var_p + diff2) / (2.0 * var_q)
        - 0.5
    ).sum(dim=-1)
    if (result < -1e-10).any():
        raise FloatingPointError("analytic Gaussian KL is materially negative")
    return result


def diagonal_gaussian_symmetric_kl(mu_a, std_a, mu_b, std_b):
    return 0.5 * (
        diagonal_gaussian_forward_kl(mu_a, std_a, mu_b, std_b)
        + diagonal_gaussian_forward_kl(mu_b, std_b, mu_a, std_a)
    )


def diagonal_gaussian_w2_sq(mu_a, std_a, mu_b, std_b):
    return (
        (mu_a.double() - mu_b.double()).square().sum(-1)
        + (std_a.double() - std_b.double()).square().sum(-1)
    )


def stable_gmm_log_prob(actions, means, stds, logits):
    actions = actions.double()
    means = means.double()
    stds = stds.double()
    logits = logits.double()
    if means.ndim == 2:
        means = means.unsqueeze(0)
        stds = stds.unsqueeze(0)
        logits = logits.unsqueeze(0)
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    component_logp = -0.5 * (
        ((actions.unsqueeze(-2) - means) / stds).square()
        + 2.0 * torch.log(stds)
        + math.log(2.0 * math.pi)
    ).sum(-1)
    return torch.logsumexp(
        torch.log_softmax(logits, dim=-1) + component_logp, dim=-1
    )


def deterministic_gmm_forward_kl(
    anchor_stats, current_stats, sample_count=64, generator_seed=0
):
    means_a = anchor_stats["means"].detach().double().cpu()
    stds_a = anchor_stats["stds"].detach().double().cpu()
    logits_a = anchor_stats["logits"].detach().double().cpu()
    means_c = current_stats["means"].detach().double().cpu()
    stds_c = current_stats["stds"].detach().double().cpu()
    logits_c = current_stats["logits"].detach().double().cpu()
    if means_a.shape != means_c.shape:
        raise ValueError("anchor/current GMM shape mismatch")
    if isinstance(generator_seed, (list, tuple)):
        seeds = [int(value) for value in generator_seed]
        if len(seeds) != means_a.shape[0]:
            raise ValueError("per-record GMM seed count mismatch")
    else:
        seeds = [int(generator_seed) + index for index in range(means_a.shape[0])]
    results = []
    for index in range(means_a.shape[0]):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seeds[index] % (2**31 - 1))
        probabilities = torch.softmax(logits_a[index], dim=-1)
        components = torch.multinomial(
            probabilities,
            int(sample_count),
            replacement=True,
            generator=generator,
        )
        eps = torch.randn(
            int(sample_count), means_a.shape[-1],
            dtype=torch.float64, generator=generator,
        )
        actions = means_a[index, components] + stds_a[index, components] * eps
        log_anchor = stable_gmm_log_prob(
            actions, means_a[index], stds_a[index], logits_a[index]
        )
        log_current = stable_gmm_log_prob(
            actions, means_c[index], stds_c[index], logits_c[index]
        )
        results.append((log_anchor - log_current).mean())
    return torch.stack(results)


def quantile(values, q=0.9):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("empty quantile input")
    return float(torch.quantile(values, float(q), interpolation="linear"))


def aggregate_phase_metric(values, bins):
    values = [float(value) for value in values]
    bins = [int(value) for value in bins]
    if len(values) != len(bins) or not values:
        raise ValueError("phase values/bins mismatch")
    means = []
    for bin_id in range(5):
        selected = [value for value, found in zip(values, bins) if found == bin_id]
        if not selected:
            raise ValueError(f"phase bin {bin_id} is empty")
        means.append(statistics.mean(selected))
    return {
        "mean": statistics.mean(values),
        "bin0": means[0],
        "bin1": means[1],
        "bin2": means[2],
        "bin3": means[3],
        "bin4": means[4],
        "worst_bin": max(means),
        "late_bin": means[4],
    }


def _key(row):
    return (
        str(row["method"]), int(row["seed"]), int(row["stage"]),
        int(row["epoch"]), int(row["task"]),
    )


def _unique_index(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate signal key: {key}")
        result[key] = row
    return result


def matched_signal_contrasts(signal_rows, rollout_rows, signal, signal_family):
    signal_index = _unique_index(signal_rows)
    rollout_index = _unique_index(
        row for row in rollout_rows if int(row["task"]) < int(row["stage"])
    )
    if set(signal_index) != set(rollout_index):
        raise ValueError("signal/rollout old-task keys differ")
    cases = sorted({key[1:] for key in signal_index})
    rows = []
    for seed, stage, epoch, task in cases:
        v1_key = ("temporal_v1", seed, stage, epoch, task)
        guard_key = ("zero_guard", seed, stage, epoch, task)
        v1_value = float(signal_index[v1_key][signal])
        guard_value = float(signal_index[guard_key][signal])
        advantage = v1_value - guard_value
        v1_success = float(rollout_index[v1_key]["success_pct"])
        guard_success = float(rollout_index[guard_key]["success_pct"])
        rollout_advantage = guard_success - v1_success
        non_tie = rollout_advantage != 0.0
        concordant = (
            (advantage > 0 and rollout_advantage > 0)
            or (advantage < 0 and rollout_advantage < 0)
        )
        rows.append({
            "signal": signal,
            "signal_family": signal_family,
            "seed": seed,
            "stage": stage,
            "epoch": epoch,
            "task": task,
            "v1_value": v1_value,
            "guard_value": guard_value,
            "signal_advantage": advantage,
            "v1_success_pct": v1_success,
            "guard_success_pct": guard_success,
            "rollout_advantage_pp": rollout_advantage,
            "rollout_non_tie": int(non_tie),
            "quadrant": quadrant(advantage, rollout_advantage),
            "sign_concordant": "" if not non_tie else int(concordant),
            "dangerous_mismatch": int(advantage > 0 and rollout_advantage < 0),
        })
    return rows


def signal_alignment_metrics(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("empty signal contrast")
    scores = [float(row["signal_advantage"]) for row in rows]
    outcomes = [float(row["rollout_advantage_pp"]) for row in rows]
    non_tie = [row for row in rows if int(row["rollout_non_tie"])]
    labels = [float(row["rollout_advantage_pp"]) > 0 for row in non_tie]
    non_tie_scores = [float(row["signal_advantage"]) for row in non_tie]
    counts = Counter(row["quadrant"] for row in rows)
    predicted_better = [
        row for row in non_tie if float(row["signal_advantage"]) > 0
    ]
    dangerous = [
        row for row in predicted_better
        if float(row["rollout_advantage_pp"]) < 0
    ]
    return {
        "rows": len(rows),
        "non_tie_rows": len(non_tie),
        "pearson": pearson(scores, outcomes),
        "spearman": spearman(scores, outcomes),
        "roc_auc": roc_auc(non_tie_scores, labels),
        "average_precision": average_precision(non_tie_scores, labels),
        "sign_concordance": sum(int(row["sign_concordant"]) for row in non_tie)
        / len(non_tie),
        "dangerous_rate": len(dangerous) / len(predicted_better)
        if predicted_better else float("nan"),
        "BB": counts["BB"], "BW": counts["BW"],
        "WB": counts["WB"], "WW": counts["WW"], "ties": counts["TIE"],
    }


def summarize_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row[field])].append(row)
    return [{field: key, **signal_alignment_metrics(groups[key])} for key in sorted(groups)]


def rank_primary_signals(summary_rows):
    order = {name: index for index, name in enumerate(PRIMARY_SIGNALS)}
    return sorted(
        [row for row in summary_rows if row["signal"] in order],
        key=lambda row: (
            -float(row["roc_auc"]), -float(row["spearman"]),
            -float(row["sign_concordance"]), float(row["dangerous_rate"]),
            order[row["signal"]],
        ),
    )


def _supported_metric(row):
    return (
        float(row["roc_auc"]) >= 0.70
        and float(row["spearman"]) >= 0.30
        and float(row["sign_concordance"]) >= 0.65
        and float(row["dangerous_rate"]) <= 0.30
        and float(row["delta_auc"]) >= 0.15
    )


def decide_h19(summary_rows, baseline, seed_rows, coverage_ok=True):
    primaries = [row for row in summary_rows if row["signal"] in PRIMARY_SIGNALS]
    if not coverage_ok or len(primaries) != 8:
        return "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_INCONCLUSIVE"
    for row in primaries:
        replicated = sum(
            float(seed["roc_auc"]) >= 0.60
            and float(seed["sign_concordance"]) >= 0.55
            for seed in seed_rows if seed["signal"] == row["signal"]
        )
        if _supported_metric(row) and replicated >= 2:
            return "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_SUPPORTED"
    if all(
        float(row["roc_auc"]) <= 0.60
        and float(row["spearman"]) <= 0.20
        and float(row["sign_concordance"]) <= 0.58
        for row in primaries
    ):
        return "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_NOT_SUPPORTED"
    return "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_INCONCLUSIVE"


def decide_flags(summary_rows, best_primary):
    flags = []
    by_name = {row["signal"]: row for row in summary_rows}
    policy = [by_name[name] for name in PRIMARY_SIGNALS]
    bc = [by_name[name] for name in BC_CONTROLS]
    best_policy = max(policy, key=lambda row: float(row["roc_auc"]))
    best_bc = max(bc, key=lambda row: float(row["roc_auc"]))
    if (
        float(best_policy["roc_auc"]) - float(best_bc["roc_auc"]) >= 0.10
        and float(best_policy["sign_concordance"])
        - float(best_bc["sign_concordance"]) >= 0.05
    ):
        flags.append("POLICY_DISTRIBUTION_ADDS_VALUE")
    bc_supported = any(
        float(row["roc_auc"]) >= 0.70
        and float(row["spearman"]) >= 0.30
        and float(row["sign_concordance"]) >= 0.65
        and float(row["dangerous_rate"]) <= 0.30
        for row in bc[1:]
    )
    if bc_supported and float(best_policy["roc_auc"]) - float(best_bc["roc_auc"]) < 0.05:
        flags.append("LOCALIZED_BC_ALONE_SUFFICIENT")
    name = best_primary["signal"]
    if name in ("rollout_kl_global_q90", "rollout_w2_global_q90"):
        flags.append("TAIL_SIGNAL_DOMINANT")
    if "phase_" in name:
        flags.append("TEMPORAL_SIGNAL_DOMINANT")
    full = by_name.get("full_gmm_fwdkl_global_mean")
    if full:
        delta = float(best_primary["roc_auc"]) - float(full["roc_auc"])
        if delta >= 0.05:
            flags.append("ROLLOUT_MODE_SIGNAL_DOMINANT")
        elif delta <= -0.05:
            flags.append("FULL_GMM_SIGNAL_DOMINANT")
    return flags or ["NONE"]
