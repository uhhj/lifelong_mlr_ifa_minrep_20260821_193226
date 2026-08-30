import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from libero.lifelong.analysis.retention_score_calibration import (
    average_precision,
    average_ranks,
    roc_auc,
)


METHODS = ("temporal_v1", "zero_guard")
TASKS = (6, 7, 8, 9)
EPOCHS = (70, 75, 80, 85, 90, 95)
LINE = re.compile(
    r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)"
)


def row_key(row):
    return (
        str(row["method"]),
        int(row["seed"]),
        int(row["stage"]),
        int(row["epoch"]),
        int(row["task"]),
    )


def build_unique_index(rows):
    index = {}
    for row in rows:
        key = row_key(row)
        if key in index:
            raise ValueError(f"duplicate alignment key: {key}")
        index[key] = row
    return index


def pearson(xs, ys):
    xs = [float(value) for value in xs]
    ys = [float(value) for value in ys]
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    dx = [value - mx for value in xs]
    dy = [value - my for value in ys]
    nx = math.sqrt(sum(value * value for value in dx))
    ny = math.sqrt(sum(value * value for value in dy))
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / (nx * ny)


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def quadrant(bc_advantage, success_advantage):
    bc_advantage = float(bc_advantage)
    success_advantage = float(success_advantage)
    if bc_advantage > 0 and success_advantage > 0:
        return "BB"
    if bc_advantage > 0 and success_advantage < 0:
        return "BW"
    if bc_advantage < 0 and success_advantage > 0:
        return "WB"
    if bc_advantage < 0 and success_advantage < 0:
        return "WW"
    return "TIE"


def matched_method_contrasts(bc_rows, rollout_rows):
    bc_index = build_unique_index(bc_rows)
    rollout_index = build_unique_index(rollout_rows)
    if set(bc_index) != set(rollout_index):
        missing_bc = sorted(set(rollout_index) - set(bc_index))
        missing_rollout = sorted(set(bc_index) - set(rollout_index))
        raise ValueError(
            "BC/rollout key mismatch: "
            f"missing_bc={missing_bc[:3]} missing_rollout={missing_rollout[:3]}"
        )
    cases = sorted(
        {
            (seed, stage, epoch, task)
            for method, seed, stage, epoch, task in bc_index
            if task < stage
        }
    )
    rows = []
    for seed, stage, epoch, task in cases:
        v1_key = ("temporal_v1", seed, stage, epoch, task)
        guard_key = ("zero_guard", seed, stage, epoch, task)
        if v1_key not in bc_index or guard_key not in bc_index:
            raise ValueError(
                f"matched method pair incomplete: {(seed, stage, epoch, task)}"
            )
        v1_loss = float(bc_index[v1_key]["mean_bc_loss"])
        guard_loss = float(bc_index[guard_key]["mean_bc_loss"])
        raw_advantage = v1_loss - guard_loss
        relative_advantage = raw_advantage / (abs(v1_loss) + 1e-12)
        v1_success = float(rollout_index[v1_key]["success_pct"])
        guard_success = float(rollout_index[guard_key]["success_pct"])
        rollout_advantage = guard_success - v1_success
        non_tie = rollout_advantage != 0.0
        sign_concordant = (
            (relative_advantage > 0 and rollout_advantage > 0)
            or (relative_advantage < 0 and rollout_advantage < 0)
        )
        rows.append(
            {
                "seed": seed,
                "stage": stage,
                "epoch": epoch,
                "task": task,
                "v1_bc_loss": v1_loss,
                "guard_bc_loss": guard_loss,
                "raw_bc_advantage": raw_advantage,
                "relative_bc_advantage": relative_advantage,
                "v1_success_pct": v1_success,
                "guard_success_pct": guard_success,
                "rollout_advantage_pp": rollout_advantage,
                "quadrant": quadrant(relative_advantage, rollout_advantage),
                "rollout_non_tie": int(non_tie),
                "sign_concordant": "" if not non_tie else int(sign_concordant),
                "dangerous_mismatch": int(
                    relative_advantage > 0 and rollout_advantage < 0
                ),
            }
        )
    return rows


def alignment_metrics(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize empty alignment rows")
    bc = [float(row["relative_bc_advantage"]) for row in rows]
    success = [float(row["rollout_advantage_pp"]) for row in rows]
    non_tie = [row for row in rows if float(row["rollout_advantage_pp"]) != 0.0]
    scores = [float(row["relative_bc_advantage"]) for row in non_tie]
    labels = [float(row["rollout_advantage_pp"]) > 0.0 for row in non_tie]
    counts = Counter(row["quadrant"] for row in rows)
    bc_better = [row for row in rows if float(row["relative_bc_advantage"]) > 0.0]
    bc_better_non_tie = [
        row
        for row in bc_better
        if float(row["rollout_advantage_pp"]) != 0.0
    ]
    dangerous = [
        row
        for row in bc_better_non_tie
        if float(row["rollout_advantage_pp"]) < 0.0
    ]
    return {
        "matched_rows": len(rows),
        "rollout_non_tie_rows": len(non_tie),
        "pearson": pearson(bc, success),
        "spearman": spearman(bc, success),
        "roc_auc_non_tie": roc_auc(scores, labels),
        "average_precision_non_tie": average_precision(scores, labels),
        "sign_concordance_non_tie": (
            sum(int(row["sign_concordant"]) for row in non_tie) / len(non_tie)
            if non_tie
            else float("nan")
        ),
        "BB": counts["BB"],
        "BW": counts["BW"],
        "WB": counts["WB"],
        "WW": counts["WW"],
        "ties": counts["TIE"],
        "bc_better_rows": len(bc_better),
        "bc_better_non_tie_rows": len(bc_better_non_tie),
        "dangerous_BW_rate": (
            len(dangerous) / len(bc_better_non_tie)
            if bc_better_non_tie
            else float("nan")
        ),
    }


def _summarize_group(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row[field])].append(row)
    return [
        {field: key, **alignment_metrics(groups[key])}
        for key in sorted(groups)
    ]


def summarize_by_seed(rows):
    return _summarize_group(rows, "seed")


def summarize_by_stage(rows):
    return _summarize_group(rows, "stage")


def build_longitudinal_relations(bc_rows, rollout_rows, epochs=EPOCHS):
    bc_index = build_unique_index(bc_rows)
    rollout_index = build_unique_index(rollout_rows)
    if set(bc_index) != set(rollout_index):
        raise ValueError("BC/rollout key mismatch for longitudinal analysis")
    groups = defaultdict(list)
    for key in bc_index:
        method, seed, stage, epoch, task = key
        if task < stage:
            groups[(method, seed, stage, task)].append(epoch)
    rows = []
    for (method, seed, stage, task), found_epochs in sorted(groups.items()):
        if sorted(found_epochs) != list(epochs):
            raise ValueError(
                f"relation {(method, seed, stage, task)} has epochs {sorted(found_epochs)}"
            )
        losses = [
            float(bc_index[(method, seed, stage, epoch, task)]["mean_bc_loss"])
            for epoch in epochs
        ]
        successes = [
            float(rollout_index[(method, seed, stage, epoch, task)]["success_pct"])
            for epoch in epochs
        ]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "stage": stage,
                "task": task,
                "points": len(epochs),
                "success_unique_values": len(set(successes)),
                "pearson": pearson([-value for value in losses], successes),
                "spearman": spearman([-value for value in losses], successes),
                "bc_loss_ep70": losses[0],
                "bc_loss_ep95": losses[-1],
                "success_ep70": successes[0],
                "success_ep95": successes[-1],
            }
        )
    return rows


def build_adjacent_transitions(bc_rows, rollout_rows, epochs=EPOCHS):
    bc_index = build_unique_index(bc_rows)
    rollout_index = build_unique_index(rollout_rows)
    if set(bc_index) != set(rollout_index):
        raise ValueError("BC/rollout key mismatch for transitions")
    relations = sorted(
        {
            (method, seed, stage, task)
            for method, seed, stage, epoch, task in bc_index
            if task < stage
        }
    )
    rows = []
    for method, seed, stage, task in relations:
        for start, end in zip(epochs[:-1], epochs[1:]):
            start_key = (method, seed, stage, start, task)
            end_key = (method, seed, stage, end, task)
            loss_start = float(bc_index[start_key]["mean_bc_loss"])
            loss_end = float(bc_index[end_key]["mean_bc_loss"])
            success_start = float(rollout_index[start_key]["success_pct"])
            success_end = float(rollout_index[end_key]["success_pct"])
            bc_improvement = loss_start - loss_end
            success_improvement = success_end - success_start
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "stage": stage,
                    "task": task,
                    "epoch_start": start,
                    "epoch_end": end,
                    "bc_loss_start": loss_start,
                    "bc_loss_end": loss_end,
                    "bc_improvement": bc_improvement,
                    "success_start": success_start,
                    "success_end": success_end,
                    "success_improvement_pp": success_improvement,
                    "quadrant": quadrant(bc_improvement, success_improvement),
                    "sign_concordant": (
                        ""
                        if success_improvement == 0.0
                        else int(
                            (bc_improvement > 0 and success_improvement > 0)
                            or (bc_improvement < 0 and success_improvement < 0)
                        )
                    ),
                }
            )
    return rows


def parse_selected_epochs(formal_eval_log, epochs=EPOCHS):
    records = {}
    for text in Path(formal_eval_log).read_text(encoding="utf-8").splitlines():
        match = LINE.search(text)
        if match:
            task, epoch, stage = map(int, match.group(1, 2, 3))
            if task == stage and epoch in epochs:
                records[(stage, epoch)] = float(match.group(4))
    selected = {}
    for stage in TASKS:
        missing = [epoch for epoch in epochs if (stage, epoch) not in records]
        if missing:
            raise ValueError(
                f"missing current-task epochs stage={stage}: {missing}"
            )
        selected[stage] = max(
            epochs, key=lambda epoch: (records[(stage, epoch)], epoch)
        )
    return selected


def build_nbt_surrogate_rows(bc_rows, rollout_rows, selected_epochs):
    bc_index = build_unique_index(bc_rows)
    rollout_index = build_unique_index(rollout_rows)
    if set(bc_index) != set(rollout_index):
        raise ValueError("BC/rollout key mismatch for NBT surrogate")
    rows = []
    for method in METHODS:
        for seed in sorted({key[1] for key in bc_index}):
            selected = selected_epochs[(method, seed)]
            for task in (6, 7, 8):
                acquisition_stage = task
                acquisition_epoch = selected[acquisition_stage]
                acquisition_key = (
                    method,
                    seed,
                    acquisition_stage,
                    acquisition_epoch,
                    task,
                )
                later_stages = list(range(task + 1, 10))
                later_epochs = [selected[stage] for stage in later_stages]
                later_keys = [
                    (method, seed, stage, selected[stage], task)
                    for stage in later_stages
                ]
                acquisition_loss = float(
                    bc_index[acquisition_key]["mean_bc_loss"]
                )
                acquisition_success = float(
                    rollout_index[acquisition_key]["success_pct"]
                )
                later_losses = [
                    float(bc_index[key]["mean_bc_loss"])
                    for key in later_keys
                ]
                later_successes = [
                    float(rollout_index[key]["success_pct"])
                    for key in later_keys
                ]
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "task": task,
                        "acquisition_stage": acquisition_stage,
                        "acquisition_epoch": acquisition_epoch,
                        "acquisition_bc_loss": acquisition_loss,
                        "acquisition_success_pct": acquisition_success,
                        "later_stages": "+".join(map(str, later_stages)),
                        "later_selected_epochs": "+".join(map(str, later_epochs)),
                        "mean_later_bc_loss": statistics.mean(later_losses),
                        "mean_later_success_pct": statistics.mean(later_successes),
                        "bc_forgetting": statistics.mean(later_losses)
                        - acquisition_loss,
                        "rollout_forgetting_pp": acquisition_success
                        - statistics.mean(later_successes),
                    }
                )
    return rows


def build_nbt_method_contrasts(nbt_rows):
    index = {
        (str(row["method"]), int(row["seed"]), int(row["task"])): row
        for row in nbt_rows
    }
    if len(index) != len(nbt_rows):
        raise ValueError("duplicate NBT surrogate row")
    rows = []
    for seed in sorted({key[1] for key in index}):
        for task in (6, 7, 8):
            v1 = index[("temporal_v1", seed, task)]
            guard = index[("zero_guard", seed, task)]
            bc_advantage = float(v1["bc_forgetting"]) - float(
                guard["bc_forgetting"]
            )
            rollout_advantage = float(v1["rollout_forgetting_pp"]) - float(
                guard["rollout_forgetting_pp"]
            )
            non_tie = rollout_advantage != 0.0
            rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "v1_bc_forgetting": float(v1["bc_forgetting"]),
                    "guard_bc_forgetting": float(guard["bc_forgetting"]),
                    "bc_retention_advantage": bc_advantage,
                    "v1_rollout_forgetting_pp": float(
                        v1["rollout_forgetting_pp"]
                    ),
                    "guard_rollout_forgetting_pp": float(
                        guard["rollout_forgetting_pp"]
                    ),
                    "rollout_retention_advantage_pp": rollout_advantage,
                    "quadrant": quadrant(bc_advantage, rollout_advantage),
                    "rollout_non_tie": int(non_tie),
                    "sign_concordant": (
                        ""
                        if not non_tie
                        else int(
                            (bc_advantage > 0 and rollout_advantage > 0)
                            or (bc_advantage < 0 and rollout_advantage < 0)
                        )
                    ),
                    "dangerous_mismatch": int(
                        bc_advantage > 0 and rollout_advantage < 0
                    ),
                }
            )
    return rows


def decide_verdict(summary, seed_summaries=None, evidence_ok=True):
    if not evidence_ok:
        return "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_INCONCLUSIVE"
    seed_summaries = seed_summaries or []
    low_seed_count = sum(
        float(row["sign_concordance_non_tie"]) <= 0.65
        for row in seed_summaries
    )
    high_seed_count = sum(
        float(row["sign_concordance_non_tie"]) >= 0.65
        for row in seed_summaries
    )
    if (
        float(summary["spearman"]) <= 0.30
        and float(summary["roc_auc_non_tie"]) <= 0.60
        and float(summary["sign_concordance_non_tie"]) <= 0.60
        and float(summary["dangerous_BW_rate"]) >= 0.30
        and low_seed_count >= 2
    ):
        return "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_SUPPORTED"
    if (
        float(summary["spearman"]) >= 0.60
        and float(summary["roc_auc_non_tie"]) >= 0.70
        and float(summary["sign_concordance_non_tie"]) >= 0.70
        and float(summary["dangerous_BW_rate"]) <= 0.20
        and high_seed_count >= 2
    ):
        return "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_NOT_SUPPORTED"
    return "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_INCONCLUSIVE"


def decide_flags(summary, seed_summaries, longitudinal_rows, nbt_contrasts):
    flags = []
    valid_longitudinal = [
        float(row["spearman"])
        for row in longitudinal_rows
        if math.isfinite(float(row["spearman"]))
    ]
    if valid_longitudinal and statistics.median(valid_longitudinal) <= 0.30:
        flags.append("LONGITUDINAL_BC_ROLLOUT_ALIGNMENT_WEAK")
    nbt_non_tie = [row for row in nbt_contrasts if int(row["rollout_non_tie"])]
    nbt_concordance = (
        sum(int(row["sign_concordant"]) for row in nbt_non_tie)
        / len(nbt_non_tie)
        if nbt_non_tie
        else float("nan")
    )
    if sum(int(row["dangerous_mismatch"]) for row in nbt_contrasts) >= 3 or (
        math.isfinite(nbt_concordance) and nbt_concordance <= 0.55
    ):
        flags.append("FORMAL_NBT_SURROGATE_MISMATCH_OBSERVED")
    by_seed = {int(row["seed"]): row for row in seed_summaries}
    if set(by_seed) == {42, 43, 44}:
        baseline_concordance = statistics.mean(
            float(by_seed[seed]["sign_concordance_non_tie"])
            for seed in (42, 43)
        )
        baseline_spearman = statistics.mean(
            float(by_seed[seed]["spearman"]) for seed in (42, 43)
        )
        if (
            float(by_seed[44]["sign_concordance_non_tie"])
            - baseline_concordance
            >= 0.15
            and float(by_seed[44]["spearman"]) > baseline_spearman
        ):
            flags.append("SEED44_BC_ROLLOUT_ALIGNMENT_DISTINCT")
    return flags
