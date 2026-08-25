# MLR + Static Robot-IGFA on LIBERO-OBJECT

## Identity

- Verdict: `STATIC_ROBOT_IGFA_INCONCLUSIVE`
- Branch: `research/static-robot-igfa`
- Starting SHA: `d928266719d0c4f7c9540c9cd5b24f0e238cf377`
- Final SHA: the branch HEAD containing this report (reported in the Git handoff; a commit cannot contain its own SHA)
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB; driver 535.230.02
- Environment: Python 3.8.20, PyTorch 1.13.0+cu116, CUDA runtime 11.6, MuJoCo 3.2.3, robosuite 1.4.0, transformers 4.26.1

The result uses three project reproducibility seeds (42, 43, 44), not claimed paper-author seeds. Each seed completed tasks 6-9 with 100 epochs per stage, batch size 10, AdamW at initial learning rate 1e-4, the unchanged MLR replay pipeline, and the unchanged 20-trial released evaluation protocol.

## Method

`MLR + Static Robot-IGFA` changes only the current-task policy-head gradient on mixed current/replay batches. It computes a single aggregate replay-anchor gradient and applies a signed projection iff the global policy-head dot product is negative:

```text
g_new <- g_new - (g_new dot g_replay / ||g_replay||^2) g_replay
```

- Scope: policy head only
- Gate: fires iff `dot(g_new, g_replay) < 0`
- Strength: alpha = 1.0
- Replay anchor: all replay samples in the current mixed batch, aggregated once
- Weighting: exact baseline fractions `n_new / B` and `n_replay / B`
- Transformer: unchanged baseline MLR mixed-batch gradient
- IFA: disabled
- Replay, features, optimizer, scheduler, epochs, batch size, learning rate, grad clipping, checkpoint selection, and evaluation: unchanged

This is a policy-head functional gate adapted to the released MLR pipeline. It is not claimed as a line-by-line reproduction of an original IGFA implementation, and it contains no temporal/progress/behavior conditioning.

## Parity and smoke

Five focused unit tests passed. On one real seed42 mixed batch with 6 current and 4 replay samples, setting `gate_strength=0` produced a maximum absolute head-gradient difference of `1.907348633e-06` and relative L2 difference of `1.546178825e-07` versus baseline `loss_total.backward()`. This passes the fixed tolerances of `5e-6` and `1e-6`, respectively.

The two-epoch seed42 smoke over tasks 6-7 passed forward/backward, checkpoint writing, and gate-stat persistence. Stage7 contained 442 paired batches, 157 conflicts, a 35.52% fire rate, mean conflict cosine -0.1545, and zero zero-norm replay anchors. Smoke-only artifacts were removed before formal execution. See `smoke_validation.log`.

## Formal results

Metrics are percentages. Standard deviations are sample standard deviations across seeds.

| Seed | FWT | NBT | AUC | Delta FWT vs MLR | Delta NBT vs MLR | Delta AUC vs MLR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 81.250 | -0.556 | 81.771 | +5.000 | -11.667 | +11.042 |
| 43 | 76.250 | 8.611 | 73.646 | -1.250 | +15.833 | -7.916 |
| 44 | 71.250 | -4.167 | 73.438 | -7.500 | -8.611 | -3.125 |

| Method | FWT | NBT | AUC |
| --- | ---: | ---: | ---: |
| Frozen MLR | 77.500 +/- 1.250 | 2.778 +/- 9.279 | 76.284 +/- 5.422 |
| Frozen MLR+IFA | 78.333 +/- 0.722 | 6.852 +/- 8.608 | 74.861 +/- 3.898 |
| MLR+Static Robot-IGFA | 76.250 +/- 5.000 | 1.296 +/- 6.587 | 76.285 +/- 4.752 |

Mean same-seed delta versus frozen MLR:

- Delta FWT: -1.250 +/- 6.250 pp
- Delta NBT: -1.481 +/- 15.072 pp
- Delta AUC: +0.0004 +/- 9.858 pp
- AUC improves in 1/3 seeds

The preregistered `SUPPORTED` rule requires mean Delta AUC >= +1 pp, mean Delta NBT <= -1 pp, mean Delta FWT >= -1 pp, and AUC improvement in at least 2/3 seeds. The observed result fails the AUC, FWT, and replication requirements. The preregistered `NOT_SUPPORTED` combination requires mean Delta AUC <= 0 plus either no NBT improvement or an FWT drop greater than 2 pp; the unrounded mean Delta AUC is slightly positive and the mean NBT improves. The rule therefore yields `STATIC_ROBOT_IGFA_INCONCLUSIVE` without post-hoc threshold changes.

## Success matrices and per-task behavior

Rows are completed stages 6-9 and columns are tasks 6-9; blank entries are not-yet-learned tasks.

Seed42:

| Stage | Task6 | Task7 | Task8 | Task9 |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 45 |  |  |  |
| 7 | 90 | 90 |  |  |
| 8 | 30 | 90 | 100 |  |
| 9 | 50 | 70 | 100 | 90 |

Seed43:

| Stage | Task6 | Task7 | Task8 | Task9 |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 20 |  |  |  |
| 7 | 65 | 100 |  |  |
| 8 | 20 | 100 | 100 |  |
| 9 | 10 | 95 | 65 | 85 |

Seed44:

| Stage | Task6 | Task7 | Task8 | Task9 |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 55 |  |  |  |
| 7 | 70 | 85 |  |  |
| 8 | 55 | 90 | 100 |  |
| 9 | 55 | 95 | 100 | 45 |

Task FWT values are `(45, 90, 100, 90)` for seed42, `(20, 100, 100, 85)` for seed43, and `(55, 85, 100, 45)` for seed44. Task6 retains the previously documented adaptation bottleneck and is not isolated as a gate failure. The main seed43 retention loss is task8 after stage9 (100 to 65; 35 pp forgetting). Seed44 preserves tasks 6-8 but has low task9 FWT (45), showing that its lower FWT is plasticity rather than old-task forgetting. Full rows are in `per_stage_success.csv`, `per_task_fwt.csv`, and `per_task_forgetting.csv`.

## Gate mechanism

Across all formal batches:

| Statistic | Value |
| --- | ---: |
| Total batches | 316,500 |
| Paired batches | 266,438 |
| Paired-batch fraction | 84.183% |
| Conflict / fired batches | 98,546 |
| Gate fire rate among paired batches | 36.987% |
| Mean conflict cosine | -0.1589 |
| Mean projection cost | 0.0588 |
| Mean current head-gradient norm | 712.35 |
| Mean replay head-gradient norm | 1,616.59 |
| Zero-anchor count | 0 |

Stage6 correctly has no paired batches. Across seeds, stage7-9 fire rates range from 34.63% to 39.28%, mean conflict cosines from -0.1465 to -0.1723, and mean projection costs from 0.0554 to 0.0630. The gate is active, stable across seeds, numerically bounded, and not dominated by zero anchors. The mechanism statistics do not explain the highly seed-dependent performance direction by a gross implementation failure. With only three seeds, no causal correlation between gate statistics and performance is claimed.

## Interpretation

Observed:

- All three seeds completed 4 x 100-epoch training and the full 30-record, 20-trial released evaluation protocol.
- The gate fires on 36.99% of paired batches with zero zero-norm replay anchors.
- Mean NBT improves by 1.48 pp, mean FWT declines by 1.25 pp, and mean AUC is effectively unchanged relative to MLR.
- Performance direction is highly seed-dependent: seed42 improves all headline metrics, while seeds43 and 44 lose AUC for different per-task reasons.

Inferred:

- Aggregate signed policy-head gating may alter the stability-plasticity balance, but it does not provide a reliable practical improvement under this three-seed protocol.
- The small, stable mechanism statistics alongside divergent outcomes are consistent with an aggregate replay anchor being too coarse, but they do not prove that explanation.

Not established:

- Static Robot-IGFA is not established as better than MLR or MLR+IFA.
- Temporal conditioning is not established as better; it has not been implemented or compared here.
- The result does not overturn the prior observation of temporal functional heterogeneity.
- Three seeds do not establish statistical significance or causal relationships between gate statistics and task outcomes.

## Verdict and next action

`STATIC_ROBOT_IGFA_INCONCLUSIVE`

H4 is inconclusive. Static signed gating measurably activates and slightly improves mean retention, but it does not improve mean AUC or preserve FWT consistently across seeds. Before choosing Temporal Robot-IGFA details, the next action is to use the retained gate statistics and prior temporal diagnosis to determine whether the mixed aggregate replay anchor is too coarse. No gate-strength sweep or temporal implementation is justified solely by this result.

## Exact successful commands

```bash
cd /data/lifelong_mlr_ifa_minrep
source /miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/mlr_ifa_minrep
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1

pytest -q tests/test_static_robot_igfa.py
bash scripts/run_object_mlr_static_robot_igfa_smoke.sh

SEED=42 bash scripts/run_object_mlr_static_robot_igfa.sh
SEED=42 bash test.sbatch mlr_static_igfa
SEED=43 bash scripts/run_object_mlr_static_robot_igfa.sh
SEED=43 bash test.sbatch mlr_static_igfa
SEED=44 bash scripts/run_object_mlr_static_robot_igfa.sh
SEED=44 bash test.sbatch mlr_static_igfa

python scripts/compute_static_robot_igfa_metrics.py
```

Formal runs, checkpoints, evaluation logs, and per-epoch gate statistics remain under `/data/lifelong_mlr_ifa_minrep/runs/object_seed{42,43,44}_mlr_static_igfa/`. No feature cache was regenerated. The committed report directory retains the complete success matrices, per-task diagnostics, mechanism summaries, metrics JSON, and concise smoke/parity evidence.
