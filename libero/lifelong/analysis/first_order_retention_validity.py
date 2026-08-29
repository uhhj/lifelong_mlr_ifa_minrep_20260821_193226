import math
from pathlib import Path

import torch
import torch.nn as nn

from libero.lifelong.algos.mlr_per_task_temporal_robot_igfa import (
    per_task_temporal_head_corrections,
    project_gradient_to_task_halfspaces,
)
from libero.lifelong.algos.mlr_static_robot_igfa import signed_replay_projection
from libero.lifelong.algos.mlr_temporal_robot_igfa import temporal_head_corrections
from libero.lifelong.analysis.per_task_temporal_interference import load_record_batch
from libero.lifelong.utils import torch_load_model


def tuple_dot(first, second):
    if len(first) != len(second):
        raise ValueError("gradient tuple length mismatch")
    return sum(
        (
            torch.sum(a.detach().double() * b.detach().double())
            for a, b in zip(first, second)
        ),
        start=torch.zeros((), dtype=torch.float64, device=first[0].device),
    )


def tuple_norm(first):
    return math.sqrt(max(float(tuple_dot(first, first)), 0.0))


def tuple_cosine(first, second, eps=1e-12):
    return float(tuple_dot(first, second)) / (tuple_norm(first) * tuple_norm(second) + eps)


def parameter_displacement_dot(gradients, displacement):
    return float(tuple_dot(gradients, displacement))


def classify_checkpoint_transition(pred_head, pred_transformer, actual_delta):
    pred_all = float(pred_head) + float(pred_transformer)
    actual = float(actual_delta)
    return {
        "pred_all": pred_all,
        "head_false_safe": bool(pred_head <= 0.0 and actual > 0.0),
        "checkpoint_transformer_flip": bool(pred_head <= 0.0 and pred_all > 0.0),
        "full_first_order_false_safe": bool(pred_all <= 0.0 and actual > 0.0),
        "sign_agreement": bool(
            (pred_all > 0.0 and actual > 0.0)
            or (pred_all < 0.0 and actual < 0.0)
            or (pred_all == 0.0 and actual == 0.0)
        ),
        "absolute_prediction_error": abs(pred_all - actual),
    }


def weighted_chunk_mean(chunk_means, chunk_sizes):
    total = sum(int(size) for size in chunk_sizes)
    if total <= 0 or len(chunk_means) != len(chunk_sizes):
        raise ValueError("invalid weighted chunks")
    return sum(float(value) * int(size) for value, size in zip(chunk_means, chunk_sizes)) / total


def gradients_or_zero(loss, parameters, retain_graph=False):
    raw = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, raw)
    )


def backend_parameter_groups(policy):
    head = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if name.startswith("policy_head.") and parameter.requires_grad
    ]
    transformer = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if name.startswith("temporal_transformer.") and parameter.requires_grad
    ]
    if not head or not transformer:
        raise RuntimeError("missing trainable policy-head or temporal-transformer parameters")
    return head, transformer


def load_checkpoint_into_policy(policy, checkpoint):
    state_dict = torch_load_model(str(checkpoint), map_location="cpu")[0]
    incompatible = policy.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility for {checkpoint}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    policy.eval()


def capture_backend_parameters(policy):
    head, transformer = backend_parameter_groups(policy)
    return {
        "head_names": [name for name, _ in head],
        "transformer_names": [name for name, _ in transformer],
        "head": tuple(parameter.detach().cpu().clone() for _, parameter in head),
        "transformer": tuple(parameter.detach().cpu().clone() for _, parameter in transformer),
    }


def backend_displacement(start, end):
    if start["head_names"] != end["head_names"]:
        raise ValueError("policy-head parameter order changed")
    if start["transformer_names"] != end["transformer_names"]:
        raise ValueError("transformer parameter order changed")
    return {
        "head": tuple(after - before for before, after in zip(start["head"], end["head"])),
        "transformer": tuple(
            after - before
            for before, after in zip(start["transformer"], end["transformer"])
        ),
    }


def _load_feature_paths(feature_paths, device):
    features = []
    actions = []
    for path in feature_paths:
        saved = torch.load(str(path), map_location="cpu")
        feature = saved["feature"]
        action = saved["actions"]
        while feature.ndim > 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        while action.ndim > 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        if tuple(feature.shape) != (8, 4, 512) or tuple(action.shape) != (8, 7):
            raise ValueError(f"unexpected latent record shape: {path}")
        features.append(feature)
        actions.append(action)
    return torch.stack(features).to(device), torch.stack(actions).to(device)


def _loss_vector(policy, features, actions):
    temporal = policy.temporal_encode(features)
    distribution = policy.policy_head(temporal[:, -1, :], train=True)
    losses = policy.policy_head.loss_fn(distribution, actions, reduction="none")
    if not torch.isfinite(losses).all():
        raise FloatingPointError("non-finite old-task probe loss")
    return losses


def mean_loss_and_group_gradients(policy, feature_paths, batch_size=32):
    """Return sample-weighted mean loss and CPU gradients for both backend groups."""
    policy.eval()
    device = next(policy.parameters()).device
    head, transformer = backend_parameter_groups(policy)
    parameters = [parameter for _, parameter in head + transformer]
    accumulators = [torch.zeros_like(parameter, device="cpu") for parameter in parameters]
    loss_sum = 0.0
    sample_count = 0
    paths = [Path(path) for path in feature_paths]
    for start in range(0, len(paths), int(batch_size)):
        chunk = paths[start : start + int(batch_size)]
        features, actions = _load_feature_paths(chunk, device)
        losses = _loss_vector(policy, features, actions)
        gradients = gradients_or_zero(losses.mean(), parameters)
        count = len(chunk)
        loss_sum += float(losses.detach().double().sum().cpu())
        sample_count += count
        for accumulator, gradient in zip(accumulators, gradients):
            accumulator.add_(gradient.detach().cpu(), alpha=count)
        del features, actions, losses, gradients
    if sample_count != len(paths) or not sample_count:
        raise RuntimeError("old-task probe sampling mismatch")
    means = tuple(value / sample_count for value in accumulators)
    return {
        "mean_loss": loss_sum / sample_count,
        "head_gradient": means[: len(head)],
        "transformer_gradient": means[len(head) :],
    }


@torch.no_grad()
def evaluate_mean_loss(policy, feature_paths, batch_size=32):
    policy.eval()
    device = next(policy.parameters()).device
    loss_sum = 0.0
    sample_count = 0
    paths = [Path(path) for path in feature_paths]
    for start in range(0, len(paths), int(batch_size)):
        chunk = paths[start : start + int(batch_size)]
        features, actions = _load_feature_paths(chunk, device)
        losses = _loss_vector(policy, features, actions)
        loss_sum += float(losses.detach().double().sum().cpu())
        sample_count += len(chunk)
    return loss_sum / sample_count


def simulate_layer_a_batch(
    policy,
    batch_records,
    current_task,
    method,
    gate_strength=1.0,
    n_bins=5,
    grad_clip=100.0,
):
    """Simulate a V1/V2 observe call through clipping, without optimizer.step."""
    policy.eval()
    policy.zero_grad(set_to_none=True)
    device = next(policy.parameters()).device
    features, actions, task_ids, progress_bins = load_record_batch(batch_records, device)
    losses = _loss_vector(policy, features, actions)
    head_named, transformer_named = backend_parameter_groups(policy)
    head = [parameter for _, parameter in head_named]
    transformer = [parameter for _, parameter in transformer_named]
    current_mask = task_ids == int(current_task)
    replay_mask = ~current_mask
    resolved = progress_bins >= 0
    relation_cache = []

    for bin_id in range(int(n_bins)):
        new_mask = current_mask & resolved & (progress_bins == bin_id)
        replay_bin = replay_mask & resolved & (progress_bins == bin_id)
        if not bool(new_mask.any()) or not bool(replay_bin.any()):
            continue
        g_new_head = gradients_or_zero(losses[new_mask].mean(), head, retain_graph=True)
        old_tasks = sorted(int(value) for value in torch.unique(task_ids[replay_bin]).tolist())
        old = []
        for old_task in old_tasks:
            old_mask = replay_bin & (task_ids == old_task)
            both = gradients_or_zero(
                losses[old_mask].mean(), head + transformer, retain_graph=True
            )
            old.append(
                {
                    "task_id": old_task,
                    "head": tuple(value.detach().clone() for value in both[: len(head)]),
                    "transformer": tuple(value.detach().clone() for value in both[len(head) :]),
                    "n_samples": int(old_mask.sum().item()),
                }
            )
        if method == "V1":
            replay_head = gradients_or_zero(
                losses[replay_bin].mean(), head, retain_graph=True
            )
            projected, _ = signed_replay_projection(
                g_new_head, replay_head, gate_strength=gate_strength
            )
        elif method == "V2":
            projected, _ = project_gradient_to_task_halfspaces(
                g_new_head,
                [
                    {
                        "task_id": item["task_id"],
                        "gradient": item["head"],
                        "n_samples": item["n_samples"],
                    }
                    for item in old
                ],
                gate_strength=gate_strength,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        for item in old:
            relation_cache.append(
                {
                    "bin": bin_id,
                    "old_task": item["task_id"],
                    "projected": tuple(value.detach().clone() for value in projected),
                    "old_head": item["head"],
                    "old_transformer": item["transformer"],
                }
            )

    if method == "V1":
        correction, _ = temporal_head_corrections(
            losses, task_ids, progress_bins, current_task, head,
            gate_strength=gate_strength, n_bins=n_bins,
        )
    else:
        correction, _ = per_task_temporal_head_corrections(
            losses, task_ids, progress_bins, current_task, head,
            gate_strength=gate_strength, n_bins=n_bins,
        )
    losses.mean().backward()
    for parameter, delta in zip(head, correction):
        parameter.grad.add_(delta)
    clip_norm = nn.utils.clip_grad_norm_(policy.parameters(), float(grad_clip))
    final_head = tuple(
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in head
    )
    final_transformer = tuple(
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in transformer
    )
    rows = []
    for relation in relation_cache:
        projected_dot = float(tuple_dot(relation["projected"], relation["old_head"]))
        head_dot = float(tuple_dot(final_head, relation["old_head"]))
        transformer_dot = float(
            tuple_dot(final_transformer, relation["old_transformer"])
        )
        full_dot = head_dot + transformer_dot
        tolerance = 1e-6 * max(1.0, abs(projected_dot))
        rows.append(
            {
                "bin": relation["bin"],
                "old_task": relation["old_task"],
                "projected_current_dot": projected_dot,
                "final_head_dot": head_dot,
                "transformer_dot": transformer_dot,
                "full_dot": full_dot,
                "projected_current_safe": int(projected_dot >= -tolerance),
                "head_composition_leak": int(projected_dot >= -tolerance and head_dot < 0.0),
                "transformer_flip": int(head_dot >= 0.0 and full_dot < 0.0),
                "full_first_order_unsafe": int(full_dot < 0.0),
                "projected_current_cosine": tuple_cosine(
                    relation["projected"], relation["old_head"]
                ),
                "final_head_cosine": tuple_cosine(final_head, relation["old_head"]),
                "transformer_cosine": tuple_cosine(
                    final_transformer, relation["old_transformer"]
                ),
                "full_cosine": full_dot
                / (
                    math.sqrt(tuple_norm(final_head) ** 2 + tuple_norm(final_transformer) ** 2)
                    * math.sqrt(
                        tuple_norm(relation["old_head"]) ** 2
                        + tuple_norm(relation["old_transformer"]) ** 2
                    )
                    + 1e-12
                ),
                "head_grad_norm": tuple_norm(final_head),
                "transformer_grad_norm": tuple_norm(final_transformer),
                "old_head_grad_norm": tuple_norm(relation["old_head"]),
                "old_transformer_grad_norm": tuple_norm(relation["old_transformer"]),
                "pre_clip_global_norm": float(clip_norm),
            }
        )
    policy.zero_grad(set_to_none=True)
    return rows, {"head": final_head, "transformer": final_transformer}
