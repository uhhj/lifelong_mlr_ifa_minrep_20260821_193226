from collections import defaultdict
from pathlib import Path

import torch

from libero.lifelong.analysis.first_order_retention_validity import (
    backend_parameter_groups,
)


FORMAL_RHOS = (0.05, 0.10, 0.25, 0.50)


@torch.no_grad()
def set_interpolated_backend(policy, start_snapshot, end_snapshot, rho):
    """Set the trainable backend to a point on the realized checkpoint chord."""
    rho = float(rho)
    head, transformer = backend_parameter_groups(policy)
    groups = (
        (head, "head_names", "head"),
        (transformer, "transformer_names", "transformer"),
    )
    for named_parameters, names_key, values_key in groups:
        names = [name for name, _ in named_parameters]
        if names != start_snapshot[names_key] or names != end_snapshot[names_key]:
            raise ValueError(f"{values_key} parameter order changed")
        start_values = start_snapshot[values_key]
        end_values = end_snapshot[values_key]
        if len(named_parameters) != len(start_values) or len(start_values) != len(end_values):
            raise ValueError(f"{values_key} snapshot length mismatch")
        for (_, parameter), start, end in zip(named_parameters, start_values, end_values):
            parameter.copy_(
                start.to(device=parameter.device, dtype=parameter.dtype)
                + rho
                * (
                    end.to(device=parameter.device, dtype=parameter.dtype)
                    - start.to(device=parameter.device, dtype=parameter.dtype)
                )
            )


def load_probe_manifest(path, project_root=None):
    import csv

    path = Path(path).resolve()
    root = Path(project_root).resolve() if project_root else path.parents[2]
    probes = defaultdict(list)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if len(rows) != 2304:
        raise ValueError(f"expected 2304 probe records, found {len(rows)}")
    for row in rows:
        seed = int(row["seed"])
        old_task = int(row["old_task"])
        record_id = int(row["record_id"])
        recorded = Path(row["path"])
        resolved = recorded if recorded.is_file() else (
            root / "features" / f"object_seed{seed}" / str(old_task) / f"{record_id}.pt"
        )
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        probes[(seed, old_task)].append(resolved)
    for key, records in probes.items():
        if len(records) != 256:
            raise ValueError(f"probe group {key} has {len(records)} records, expected 256")
    if len(probes) != 9:
        raise ValueError(f"expected 9 seed/task probe groups, found {len(probes)}")
    return dict(probes)


def signs_agree(prediction, actual):
    prediction = float(prediction)
    actual = float(actual)
    return bool(
        (prediction > 0.0 and actual > 0.0)
        or (prediction < 0.0 and actual < 0.0)
        or (prediction == 0.0 and actual == 0.0)
    )


def classify_finite_step(probe_delta, actual_delta, first_order_false_safe=False):
    probe_delta = float(probe_delta)
    actual_delta = float(actual_delta)
    probe_false_safe = probe_delta <= 0.0 and actual_delta > 0.0
    return {
        "probe_false_safe": bool(probe_false_safe),
        "probe_false_alarm": bool(probe_delta > 0.0 and actual_delta <= 0.0),
        "probe_sign_agreement": signs_agree(probe_delta, actual_delta),
        "rescued_first_order_false_safe": bool(
            first_order_false_safe and probe_delta > 0.0
        ),
        "late_harmful_reversal": bool(probe_false_safe),
    }


def confusion_metrics(predictions, actuals):
    if len(predictions) != len(actuals) or not predictions:
        raise ValueError("predictions and actuals must be equally sized and non-empty")
    predicted_harmful = [float(value) > 0.0 for value in predictions]
    endpoint_harmful = [float(value) > 0.0 for value in actuals]
    tp = sum(pred and actual for pred, actual in zip(predicted_harmful, endpoint_harmful))
    tn = sum(not pred and not actual for pred, actual in zip(predicted_harmful, endpoint_harmful))
    fp = sum(pred and not actual for pred, actual in zip(predicted_harmful, endpoint_harmful))
    fn = sum(not pred and actual for pred, actual in zip(predicted_harmful, endpoint_harmful))
    harmful = tp + fn
    safe = tn + fp
    sensitivity = tp / harmful if harmful else float("nan")
    specificity = tn / safe if safe else float("nan")
    balanced = (
        (sensitivity + specificity) / 2.0
        if harmful and safe else float("nan")
    )
    return {
        "cases": len(predictions),
        "endpoint_harmful_rate": harmful / len(predictions),
        "false_safe_rate": fn / len(predictions),
        "false_alarm_rate": fp / len(predictions),
        "sign_agreement_rate": sum(
            signs_agree(prediction, actual)
            for prediction, actual in zip(predictions, actuals)
        ) / len(predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }
