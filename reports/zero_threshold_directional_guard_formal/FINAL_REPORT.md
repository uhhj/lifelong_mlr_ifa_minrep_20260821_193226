# Zero-Threshold Optimizer-Aware Directional Guard Formal

## Identity

- Verdict: `ZERO_THRESHOLD_DIRECTIONAL_GUARD_FORMAL_NOT_SUPPORTED`
- Flags: `none`
- Branch: `research/zero-threshold-directional-guard-formal`
- Starting SHA: `e2ed8e38e21c0f904516046c2b9be6bbd1efb249`
- Formal experiment source SHA: `fb39a8fb3831cab22dcb6c2bb51f0a7ca6228098`

## Formal protocol

LIBERO_OBJECT, seeds 42/43/44, tasks 6-9, 100 epochs/stage, batch 10,
AdamW LR 1e-4, 20 rollout trials, Temporal V1 five-bin gate, IFA disabled.
The directional guard uses fresh old gradients every update, tau=0, scale=0.5,
and a deterministic 64-record/task subset of the existing replay buffer.
Formal score probes are a deterministic subset of the existing 50% replay
memory, not additional old-task data.

## Main metrics

| Method | FWT | NBT ↓ | AUC |
|---|---:|---:|---:|
| MLR | 77.500 | 2.778 | 76.284 |
| Temporal V1 | 80.833 | 5.741 | 78.472 |
| Zero Guard | 75.833 ± 6.292 | 8.611 ± 4.673 | 71.806 ± 7.821 |

## Per seed

| Seed | Guard FWT | Guard NBT | Guard AUC | ΔNBT vs V1 | ΔFWT vs V1 | ΔAUC vs V1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 70.000 | 13.889 | 63.333 | +6.667 | -10.000 | -13.542 |
| 43 | 75.000 | 5.000 | 73.333 | +11.944 | -3.750 | -9.375 |
| 44 | 82.500 | 6.944 | 78.750 | -10.000 | -1.250 | +2.917 |

## Selected epochs

| Seed | Stage 6 | Stage 7 | Stage 8 | Stage 9 |
|---:|---:|---:|---:|---:|
| 42 | 80 | 85 | 75 | 80 |
| 43 | 90 | 95 | 95 | 85 |
| 44 | 95 | 80 | 95 | 90 |


## Per-task forgetting

Complete values are in `per_task_forgetting.csv`. The largest observed
forgetting was seed 42 task 7 at
25.000 percentage points.

## Guard mechanism

Across stages 7-9 the guard fired on 0.377684 of
269400 eligible updates. Stage 6 had no old-task
probe and zero eligible guard updates. All stage-level mechanism values were
finite; see `guard_stage_summary.csv`.

## Runtime

Per-seed training wall time and total observed GPU-hours are recorded in
`runtime_summary.csv`. Total observed training GPU-hours:
5.446.

## Observed

The formal rollout metrics, selected checkpoints, per-task trajectories, and
guard fire statistics above were produced by the frozen three-seed protocol.

## Inferred

H17 is classified as `ZERO_THRESHOLD_DIRECTIONAL_GUARD_FORMAL_NOT_SUPPORTED` by the preregistered NBT, FWT, and AUC rules.
This determines whether the H16 short-horizon BC-loss retention benefit
translated into formal rollout retention without post-result tuning.

## Not established

Fresh old-task gradients are still recomputed every update; formal support
would establish efficacy, not efficiency.

The method currently uses 64 replay samples per old task as a fixed
score-gradient subset.

The score probes are within the existing replay memory, so no additional
old-task dataset access is introduced.

The guard full-commits AdamW moments while scaling only the parameter endpoint.

Three seeds establish replication in this benchmark, not universal
cross-benchmark generalization.

LIBERO_OBJECT success does not yet establish GOAL/LIBERO-50 transfer.
