# Final paper-text-faithful protocol

## Scope

LIFELONG-STAGE PAPER-TEXT-FAITHFUL REPRODUCTION WITH RELEASED/FROZEN PRETRAINED INITIALIZATION

## Paper-explicit fields

- LIBERO-OBJECT; seeds 42/43/44; batch 10; 100 epochs per lifelong stage.
- AdamW, initial LR 0.0001, linear scheduler.
- Replay equivalent to five demonstrations per task (`past_data_rate=0.5`).
- IFA lambda 0.1; OBJECT alpha/gamma 0.3.
- 20 evaluation trials per seed.

## Frozen paper-unspecified choices

- Scheduler advances after every optimizer step; total steps = epochs × loader length. **PAPER DOES NOT SPECIFY SCHEDULER STEP UNIT.**
- Warmup 500: **RELEASE_FALLBACK**.
- AdamW betas [0.9, 0.999], eps 1e-08, weight decay 0.1: **RELEASE_FALLBACK**.
- Existing seed-specific augmented feature caches: **RELEASE_FALLBACK; NOT PAPER-SPECIFIED**. Features will not be regenerated.
- GMM stochastic sampling: **RELEASE_IMPLEMENTATION_FALLBACK**.
- Primary checkpoint: ep99 only, the final state after 100 zero-indexed epochs.
- Secondary: candidates 70/75/80/85/90/95/99, best current-task checkpoint with later ties, then old tasks at that checkpoint.
- Structural-collapse diagnostic: the same current task at or below 10 percentage points in at least two of three seeds for either method.

## Initialization and boundary

All runs share `object_multitask_model_ep99.pth`. Exact paper-time pretraining is not publicly recoverable, so this is not a full end-to-end exact-paper reproduction. Provenance remains `PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE`.

Starting SHA: `1f4e45d604be7f2b61b356e70ac4fa7c80557fe0`.
