# IFA optimization-regime interaction diagnosis

## Identity

- Starting SHA: `865bae54fe6ed0fde17c6e2f4656dc86270e16ec`
- Branch: `research/ifa-optimization-regime-interaction`
- Scope: LIBERO_OBJECT, seeds 42/43/44.

## Frozen existing results

Legacy MLR is 77.500 FWT / 2.778 NBT / 76.284 AUC; legacy IFA gamma0.31 is 78.333 / 6.852 / 74.861. Batch-step MLR is 90.000 / 13.704 / 82.882; batch-step IFA gamma0.30 is 86.667 / 13.611 / 79.757.

## Why the gamma0.31 bridge was required

The existing scheduler comparison was confounded by gamma 0.31 versus 0.30. The bridge changes only gamma within the batch-step IFA protocol.

## Bridge protocol

Batch 10, batch-step linear schedule with 500 warmup steps, AdamW 1e-4, replay rate 0.5, IFA weight 0.1, gamma 0.31, tasks 6-9, 100 epochs per stage, and 20 rollout trials.

## Bridge 3-seed results

- FWT: 90.417 ± 0.722
- NBT: 12.963 ± 7.769
- AUC: 83.958 ± 4.350

## Gamma0.30 vs 0.31

Gamma0.31 minus gamma0.30 is 3.750 FWT, -0.648 NBT, and 4.202 AUC. Flag: **GAMMA_030_031_MATERIAL**.

## Fixed-gamma IFA effects

Legacy IFA minus MLR: 0.833 FWT / 4.074 NBT / -1.423 AUC. Batch-step IFA minus MLR: 0.417 / -0.741 / 1.076.

## Difference-in-differences

Batch-step minus legacy relative IFA effect: -0.417 ± 1.909 FWT / -4.815 ± 1.313 NBT / 2.499 ± 3.031 AUC (sample standard deviation across seeds).

## IFA pressure

Offline pressure is reported in `pressure/`; increased-under-strong-optimization flag is False. Pressure-versus-rollout correlations use only nine seed-stage points and are descriptive.

## Stage/task decomposition

The largest mean batch-step current-task penalty occurs at stage 8 (-3.333 percentage points). Stage-specific means are stage6=0.000, stage7=0.000, stage8=-3.333, stage9=5.000. Stable old-task compensation for every batch-step stage with a current-task penalty is False. The complete current-task and old-task deltas are in `interaction/stage_current_task_effects.csv` and `interaction/stage_old_task_effects.csv`; no retention tradeoff is inferred without corresponding old-task improvement.

## Observed

The formal bridge isolates gamma 0.31 under the batch-step optimization regime. Flags: GAMMA_030_031_MATERIAL, IFA_RETENTION_GAIN_NOT_ESTABLISHED.

## Inferred

The primary rule-based verdict is **OPTIMIZATION_REGIME_IFA_INTERACTION_NOT_SUPPORTED**. This verdict concerns optimization-regime dependence in the released implementation.

## Not established

This experiment does not recover the unpublished exact internal code version used to produce the paper table.

A scheduler interaction does not prove IFA is intrinsically ineffective; it establishes optimization-regime dependence in this released implementation.

The offline pressure probe is diagnostic, not a causal substitute for the formal bridge runs.

Only LIBERO_OBJECT and seeds42/43/44 are studied.

## Verdict

**OPTIMIZATION_REGIME_IFA_INTERACTION_NOT_SUPPORTED**
