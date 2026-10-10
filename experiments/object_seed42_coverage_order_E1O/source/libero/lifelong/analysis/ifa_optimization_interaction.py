"""Pure helpers for the optimization-regime × IFA diagnosis."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


METRICS = ("fwt", "nbt", "auc")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping], fields: Optional[Sequence[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_existing_rows(root: Path) -> List[Dict]:
    root = Path(root)
    legacy_path = root / "reports/baseline_reconciliation/object_3seed_metrics.json"
    batch_mlr_path = root / "reports/paper_reproduction_v2/mlr_formal/three_seed_summary.json"
    batch_ifa_path = root / "reports/paper_reproduction_v2/ifa_formal/three_seed_summary.json"
    legacy = read_json(legacy_path)["per_seed"]
    batch_mlr = read_json(batch_mlr_path)["per_seed"]
    batch_ifa = read_json(batch_ifa_path)["per_seed"]
    rows: List[Dict] = []
    specs = (
        ("legacy_epoch", "mlr", "", legacy["mlr"], legacy_path),
        ("legacy_epoch", "ifa", "0.31", legacy["mlr_ifa"], legacy_path),
        ("batch_step", "mlr", "", batch_mlr, batch_mlr_path),
        ("batch_step", "ifa", "0.30", batch_ifa, batch_ifa_path),
    )
    for regime, method, gamma, source, source_path in specs:
        for seed in (42, 43, 44):
            item = source[str(seed)]
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "gamma": gamma,
                    "seed": seed,
                    "FWT": float(item["fwt"]),
                    "NBT": float(item["nbt"]),
                    "AUC": float(item["auc"]),
                    "source_path": str(source_path.relative_to(root)),
                }
            )
    if len(rows) != 12:
        raise AssertionError(f"expected 12 canonical rows, found {len(rows)}")
    return rows


def indexed(rows: Iterable[Mapping]) -> Dict[tuple, Mapping]:
    return {(row["regime"], row["method"], int(row["seed"])): row for row in rows}


def relative_effects(rows: Sequence[Mapping], regimes=("legacy_epoch", "batch_step")) -> List[Dict]:
    by_key = indexed(rows)
    output = []
    for regime in regimes:
        for seed in (42, 43, 44):
            base = by_key[(regime, "mlr", seed)]
            ifa = by_key[(regime, "ifa", seed)]
            output.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "ifa_gamma": ifa["gamma"],
                    "delta_fwt": float(ifa["FWT"]) - float(base["FWT"]),
                    "delta_nbt": float(ifa["NBT"]) - float(base["NBT"]),
                    "delta_auc": float(ifa["AUC"]) - float(base["AUC"]),
                }
            )
    return output


def interaction_rows(relative_rows: Sequence[Mapping], confound="") -> List[Dict]:
    lookup = {(row["regime"], int(row["seed"])): row for row in relative_rows}
    output = []
    for seed in (42, 43, 44):
        legacy = lookup[("legacy_epoch", seed)]
        batch = lookup[("batch_step", seed)]
        output.append(
            {
                "seed": seed,
                "interaction_fwt": float(batch["delta_fwt"]) - float(legacy["delta_fwt"]),
                "interaction_nbt": float(batch["delta_nbt"]) - float(legacy["delta_nbt"]),
                "interaction_auc": float(batch["delta_auc"]) - float(legacy["delta_auc"]),
                "interpretation": confound,
            }
        )
    return output


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def gamma_materiality(gamma_rows: Sequence[Mapping]) -> bool:
    fwt = [float(row["gamma_delta_fwt"]) for row in gamma_rows]
    auc = [float(row["gamma_delta_auc"]) for row in gamma_rows]

    def same_sign(values):
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        return max(positive, negative) >= 2

    return (
        abs(statistics.mean(fwt)) >= 2.0 and same_sign(fwt)
    ) or (
        abs(statistics.mean(auc)) >= 2.0 and same_sign(auc)
    )


def decide_h20(interactions: Sequence[Mapping], batch_effects: Sequence[Mapping], gamma_material: bool) -> str:
    interaction_fwt = [float(row["interaction_fwt"]) for row in interactions]
    batch_fwt = [float(row["delta_fwt"]) for row in batch_effects]
    mean_interaction = statistics.mean(interaction_fwt)
    mean_batch = statistics.mean(batch_fwt)
    supported = (
        mean_interaction <= -3.0
        and sum(value <= -1.0 for value in interaction_fwt) >= 2
        and mean_batch <= -2.0
        and not gamma_material
    )
    if supported:
        return "OPTIMIZATION_REGIME_IFA_INTERACTION_SUPPORTED"
    if abs(mean_interaction) < 1.0 and mean_batch > -1.0:
        return "OPTIMIZATION_REGIME_IFA_INTERACTION_NOT_SUPPORTED"
    return "OPTIMIZATION_REGIME_IFA_INTERACTION_INCONCLUSIVE"


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else math.nan


def pressure_increased(seed_rows: Sequence[Mapping]) -> bool:
    paired = {int(row["seed"]): row for row in seed_rows}
    ratio_direction = 0
    active_direction = 0
    for seed in (42, 43, 44):
        row = paired[seed]
        legacy_ratio = float(row["legacy_ratio_median"])
        batch_ratio = float(row["batch_ratio_median"])
        if batch_ratio > 1.25 * legacy_ratio:
            ratio_direction += 1
        if float(row["batch_active_rate"]) - float(row["legacy_active_rate"]) >= 0.10:
            active_direction += 1
    return ratio_direction >= 2 or active_direction >= 2
