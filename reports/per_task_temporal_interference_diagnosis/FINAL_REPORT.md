# Same-Bin Per-Task Functional Interference Diagnosis

## Identity

- Branch: `research/per-task-temporal-interference-diagnosis`
- Starting SHA: `a2c05a240b87c39ec8aaf1d9fbadf594f80c5f9f`
- Temporal V1 scientific ancestor: `50f4eea8266257ba81c2139de13eb09432d611b9`
- Host: `dm-MS-7D70`, 2 × NVIDIA GeForce RTX 4090
- Scope: policy head only; no optimizer step and no model, buffer, feature, or checkpoint mutation
- Hypothesis H6: aggregation of old tasks within one progress bin can mask task-specific functional interference

## Migration continuity

`50f4eea` is an ancestor of the starting SHA. The starting commit contains all three workstation compatibility changes: multiprocessing `spawn`, evaluation-time augmentation guard, and relocated `MLR_CONFIG` support. Python 3.8.20, PyTorch 1.13.0+cu116, CUDA runtime 11.6, and MuJoCo 3.2.3 were retained. No dependency upgrade, migration rerun, feature extraction, or baseline rerun was performed.

## LR scheduler note

The LR result is classification **B**, not C. AdamW is initialized at `1e-4`; `LambdaLR` construction sets the active LR to exactly zero, after which scheduler steps 1/2/10/100 produce `2e-7`, `4e-7`, `2e-6`, and `2e-5`. The release advances its configured 500-step warmup once per epoch, so the historical zero was real initially but the LR is not permanently stuck. This does not affect the present optimizer-free diagnosis. The exact trace is in `lr_scheduler_check.md` and `reports/migration_followup/lr_scheduler_check.md`.

## Protocol

- Seeds: 42, 43, 44
- New tasks: 7, 8, 9
- Progress bins: 5 normalized bins
- Batch size: 10
- Primary batches: 400 per seed/stage, 3600 total
- Targeted seed-44 late batches: 400 each for new tasks 8 and 9, 800 total
- Diagnostic seed: 20260828, with stage seed `diagnostic_seed + seed×100 + new_task`
- Replay reconstruction: each old task independently samples `ceil(0.5N)` from sorted feature paths using a fresh `random.Random(seed)`, matching `update_buffer_equal_quota_10()`
- Primary checkpoints: new task 7 → stage 6 ep95; new task 8 → stage 7 ep95; new task 9 → stage 8 ep95
- Targeted checkpoints: seed44 new task 8 → stage 8 ep90; new task 9 → stage 9 ep90
- Gate simulation: exact Temporal V1 direct same-bin replay-mean anchor with strength 1.0

Ep95 is a late-stage pre-transition proxy, not the exact in-memory ep99 boundary state. Diagnostic batches are deterministic samples from the formal training distribution, not exact replay of historical training batches.

The weighted sum of individual old-task mean gradients was audited against the direct replay-bin mean. The prescribed 2/1/3-sample focused parity test passed both `max_abs <= 5e-6` and `relative_L2 <= 1e-6`. On real high-magnitude float32 gradients, the maximum relative L2 error was `2.11e-7`; absolute differences reached `1.22e-4` because one float32 ULP exceeds the requested absolute threshold at these magnitudes. The simulated gate uses the direct gradient, exactly matching Temporal V1 rather than the rounded reconstructed sum.

## Smoke

Seed42 stage8 completed 100 batches using stage7 ep95. It produced 214 same-bin candidate events and 42 multi-old candidates. Mixed-old-sign rate was 42.86%, masked-conflict rate was 41.18% among aggregate-safe multi-old candidates, and residual-conflict rate was 52.38%. Checkpoint, sidecar, replay reconstruction, batch generation, per-task gradients, raw CSV, summary, and finite-value checks passed. Smoke-only outputs were deleted.

## Formal completeness

- 6/6 focused tests passed.
- All 3 seeds × stages 7–9 × 400 primary batches completed.
- Both seed44 ep90 targeted cases completed.
- Eleven raw event CSVs were produced; no NaN/Inf was found.
- Stage7 primary candidate counts were 764, 738, and 750 for seeds 42, 43, and 44. It has only old task6 and is correctly excluded from multi-old cancellation denominators.
- Formal stage8/9 multi-old candidate count: 1328.

## Main stage table

Rates use multi-old same-bin candidates. Masked conflict is conditional on the aggregate relation being safe (`aggregate dot >= 0`).

| Seed | New task | Multi-old candidates | Mixed sign | Masked conflict | Residual after aggregate gate | Aggregate gate fire |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 177 | 40.11% | 37.01% | 54.80% | 28.25% |
| 42 | 9 | 259 | 41.70% | 38.17% | 55.60% | 28.19% |
| 43 | 8 | 179 | 49.72% | 42.98% | 61.45% | 32.40% |
| 43 | 9 | 267 | 46.07% | 37.70% | 55.43% | 28.46% |
| 44 | 8 | 189 | 37.04% | 32.58% | 52.91% | 30.16% |
| 44 | 9 | 257 | 40.86% | 32.79% | 52.14% | 28.79% |
| **Overall** | **8–9** | **1328** | **42.62%** | **36.70%** | **55.20%** | **29.22%** |

Of the 1328 multi-old candidates, 940 were aggregate-safe. The overall counts are 566 mixed-sign candidates, 345 masked conflicts among the 940 safe candidates, 733 residual-conflict candidates, and 388 aggregate gate fires.

## Main progress-bin table

| Bin | Multi-old candidates | Mixed sign | Masked conflict | Residual after gate | Gate fire |
|---:|---:|---:|---:|---:|---:|
| 0 | 252 | 50.00% | 43.29% | 63.10% | 34.92% |
| 1 | 265 | 50.19% | 45.76% | 63.77% | 33.21% |
| 2 | 301 | 27.91% | 23.75% | 39.20% | 20.27% |
| 3 | 247 | 40.08% | 32.81% | 47.77% | 22.27% |
| 4 | 263 | 47.15% | 43.71% | 64.26% | 36.50% |

Cancellation is not confined to one bin. Bins 0, 1, and 4 are strongest; bin 2 is weaker but still has a 39.20% residual-conflict rate.

## Per-old-task table

This table pools primary stage8/9 individual relations, including both single-old and multi-old same-bin candidates.

| Old task | Candidate relations | Pre-conflict rate | Post-conflict rate | Mean pre cosine | Mean post cosine |
|---:|---:|---:|---:|---:|---:|
| 6 | 2578 | 31.19% | 18.04% | 0.02118 | 0.06343 |
| 7 | 2463 | 38.25% | 23.95% | 0.00956 | 0.05019 |
| 8 | 1097 | 38.47% | 24.70% | 0.01210 | 0.03834 |

Aggregate projection reduces average individual conflict but does not eliminate it. Old task7 is more conflicted than task6, but is comparable to task8 rather than uniquely dominant.

## Seed44 task7 case

| Relation | Checkpoint | Candidates | Pre conflict | Post conflict | Masked participation |
|---|---|---:|---:|---:|---:|
| new8 → old7 entry | stage7 ep95 | 503 | 40.95% | 22.07% | 5.96% |
| new8 → old7 late | stage8 ep90 | 503 | 37.57% | 23.26% | 4.77% |
| new9 → old7 entry | stage8 ep95 | 323 | 32.82% | 22.29% | 6.81% |
| new9 → old7 late | stage9 ep90 | 323 | 35.60% | 21.98% | 7.74% |

Task7-specific residual conflict persists after training rather than disappearing. The strongest targeted late bin is new8 → old7 bin0, with 34.02% post-conflict (52.58% pre-conflict). For new9 → old7, the largest late post-conflict is bin2 at 26.98%. The case is consistent with task7 retention pressure, but it does not uniquely identify task7 as worse than every other old task.

## Observed

- Mixed-old signs occur in 42.62% of primary multi-old candidates.
- When the aggregate relation appears safe, 36.70% still contain an individually harmed old task.
- After applying the real Temporal V1 aggregate projection, 55.20% retain at least one negative old-task relation.
- Residual rates replicate at 55.28%, 57.85%, and 52.47% for seeds 42, 43, and 44.
- All five progress bins exhibit residual task-specific conflict.
- The aggregate gate fires in only 29.22% of multi-old candidates, substantially below the residual-conflict rate.

## Inferred

- Same-progress aggregation can cancel opposing old-task relations and hide functional interference from Temporal V1's aggregate decision.
- A single aggregate projection is not sufficient to protect every old task even when it improves the aggregate relation.
- The replicated effect provides a direct mechanistic premise for testing a per-old-task + temporal gate.

## Not established

- This diagnosis does not establish that a per-task + temporal intervention will improve NBT, FWT, or AUC.
- It does not reconstruct exact historical optimizer batches or the exact ep99 in-memory transition state.
- Normalized progress is not a semantic reach/grasp/transport/place label.
- It does not test transformer projection, IFA, a different alpha, more bins, or learned phases.
- Seed44 task7 forgetting is not causally explained by this analysis alone.

## Verdict

`WITHIN_BIN_TASK_CANCELLATION_SUPPORTED`

The prespecified thresholds are all satisfied: overall mixed-old-sign and residual rates exceed 20%, and residual conflict exceeds 15% in 3/3 seeds.

## Exact successful commands

```bash
cd /media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate /home/dm/miniconda3/envs/wzp_mlr_tmp
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
source scripts/_common.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_per_task_temporal_interference.py
bash scripts/run_per_task_temporal_interference.sh smoke
bash scripts/run_per_task_temporal_interference.sh formal
```

## Next action

Implement a strictly scoped **Per-Task + Temporal Robot-IGFA** intervention using `R(T_new, T_old, bin)` and compare it against frozen Temporal V1 under identical optimization. Do not add IFA, transformer projection, an alpha sweep, or semantic phase labels in that experiment.
