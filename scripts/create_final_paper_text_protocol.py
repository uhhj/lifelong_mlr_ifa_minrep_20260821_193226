#!/usr/bin/env python3
"""Freeze the final paper-text-faithful lifelong-stage protocol."""

import json
from datetime import datetime, timezone
from pathlib import Path


STARTING_SHA = "1f4e45d604be7f2b61b356e70ac4fa7c80557fe0"


def build_protocol(root):
    return {
        "identity": {
            "name": "final_paper_text_faithful",
            "starting_sha": STARTING_SHA,
            "scope": "LIFELONG-STAGE PAPER-TEXT-FAITHFUL REPRODUCTION WITH RELEASED/FROZEN PRETRAINED INITIALIZATION",
            "benchmark": "LIBERO_OBJECT",
        },
        "source_priority": [
            "paper_explicit_text",
            "public_release_fallback_for_unspecified_fields",
            "minimal_unit_consistent_interpretation",
            "prior_runs_for_sanity_only_not_parameter_inference",
        ],
        "paper_explicit_fields": {
            "batch_size": 10,
            "epochs_per_stage": 100,
            "optimizer": "torch.optim.AdamW",
            "initial_lr": 1e-4,
            "scheduler_name": "linear",
            "buffer_policy": "total size equivalent to five demonstrations per task",
            "past_data_rate": 0.5,
            "effective_stored_demo_equivalent": 5,
            "ifa_lambda": 0.1,
            "ifa_alpha_gamma": 0.30,
            "seeds": [42, 43, 44],
            "n_eval": 20,
        },
        "paper_unspecified_fields": [
            "scheduler_step_unit",
            "warmup_steps",
            "feature_extraction_augmentation",
            "checkpoint_selection_rule",
            "exact_evaluation_action_sampling",
            "AdamW_betas_eps_weight_decay",
        ],
        "implementation_choices": {
            "scheduler_step_unit": "after_each_optimizer_step",
            "scheduler_step_unit_provenance": "MINIMAL_UNIT_CONSISTENT_INTERPRETATION; PAPER DOES NOT SPECIFY SCHEDULER STEP UNIT",
            "warmup_steps": 500,
            "warmup_provenance": "RELEASE_FALLBACK",
            "total_steps_formula": "epochs_per_stage * len(train_loader)",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "optimizer_weight_decay": 0.1,
            "optimizer_unspecified_provenance": "RELEASE_FALLBACK",
            "feature_augmentation": "seed-controlled stochastic augmentation in frozen cache generation",
            "feature_augmentation_provenance": "RELEASE_FALLBACK; NOT PAPER-SPECIFIED",
            "feature_regeneration": False,
            "action_inference_semantics": "GMM component distribution with stochastic sampling",
            "action_inference_provenance": "RELEASE_IMPLEMENTATION_FALLBACK",
            "primary_checkpoint": 99,
            "primary_checkpoint_reason": "final state after the 100th zero-indexed epoch",
            "secondary_candidate_epochs": [70, 75, 80, 85, 90, 95, 99],
            "secondary_selection": "best current-task success; later epoch on ties; old tasks at selected checkpoint",
            "secondary_label": "SECONDARY_RELEASE_COMPATIBLE_SELECTION",
            "structural_collapse_rule": "same current task at or below 10 percentage points in at least two of three seeds for either method",
        },
        "paths": {
            "pretrained_checkpoint": str(root / "checkpoints/object_multitask_model_ep99.pth"),
            "feature_cache_seed_mapping": {
                str(seed): str(root / f"features/object_seed{seed}") for seed in (42, 43, 44)
            },
            "runs": str(root / "runs/final_paper_text_faithful"),
            "buffers": str(root / "buffers/final_paper_text_faithful"),
            "reports": str(root / "reports/final_paper_text_faithful"),
        },
        "task_order": {
            "pretrained_tasks": [0, 1, 2, 3, 4, 5],
            "lifelong_tasks": [6, 7, 8, 9],
            "start_task": 6,
            "end_task_exclusive": 10,
        },
        "checkpoint_save_epochs": [70, 75, 80, 85, 90, 95, 99],
        "methods": {
            "mlr": {"ifa_enabled": False, "gamma_explicit_but_unused": 0.30},
            "ifa": {"ifa_enabled": True, "ifa_lambda": 0.1, "gamma": 0.30},
        },
        "provenance_boundary": "PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE",
        "pretraining_limitation": "paper-time pretraining code is not public; the frozen released checkpoint is shared by all runs",
    }


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/final_paper_text_faithful"
    out.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(root)
    (out / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    explicit = protocol["paper_explicit_fields"]
    choices = protocol["implementation_choices"]
    markdown = f"""# Final paper-text-faithful protocol

## Scope

{protocol['identity']['scope']}

## Paper-explicit fields

- LIBERO-OBJECT; seeds 42/43/44; batch {explicit['batch_size']}; {explicit['epochs_per_stage']} epochs per lifelong stage.
- AdamW, initial LR {explicit['initial_lr']}, linear scheduler.
- Replay equivalent to five demonstrations per task (`past_data_rate={explicit['past_data_rate']}`).
- IFA lambda {explicit['ifa_lambda']}; OBJECT alpha/gamma {explicit['ifa_alpha_gamma']}.
- 20 evaluation trials per seed.

## Frozen paper-unspecified choices

- Scheduler advances after every optimizer step; total steps = epochs × loader length. **PAPER DOES NOT SPECIFY SCHEDULER STEP UNIT.**
- Warmup {choices['warmup_steps']}: **RELEASE_FALLBACK**.
- AdamW betas {choices['optimizer_betas']}, eps {choices['optimizer_eps']}, weight decay {choices['optimizer_weight_decay']}: **RELEASE_FALLBACK**.
- Existing seed-specific augmented feature caches: **RELEASE_FALLBACK; NOT PAPER-SPECIFIED**. Features will not be regenerated.
- GMM stochastic sampling: **RELEASE_IMPLEMENTATION_FALLBACK**.
- Primary checkpoint: ep99 only, the final state after 100 zero-indexed epochs.
- Secondary: candidates 70/75/80/85/90/95/99, best current-task checkpoint with later ties, then old tasks at that checkpoint.
- Structural-collapse diagnostic: the same current task at or below 10 percentage points in at least two of three seeds for either method.

## Initialization and boundary

All runs share `object_multitask_model_ep99.pth`. Exact paper-time pretraining is not publicly recoverable, so this is not a full end-to-end exact-paper reproduction. Provenance remains `PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE`.

Starting SHA: `{STARTING_SHA}`.
"""
    (out / "PROTOCOL.md").write_text(markdown, encoding="utf-8")
    frozen = (
        "PROTOCOL_FROZEN\n"
        f"created_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"starting_sha={STARTING_SHA}\n"
        "changes_after_formal_start_require_full_restart_from_seed42\n"
    )
    (out / "PROTOCOL_FROZEN.txt").write_text(frozen, encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
