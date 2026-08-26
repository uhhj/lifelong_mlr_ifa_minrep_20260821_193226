# Temporal Robot-IGFA V1 — LIBERO-OBJECT

## Identity

- Branch: `research/temporal-robot-igfa-v1`
- Starting SHA: `e09fc8505b7618bde4e7f2908b2201662abd8344`
- Benchmark: LIBERO-OBJECT, lifelong tasks 6–9
- Seeds: 42, 43, 44
- Scale: four stages × 100 epochs and 20 evaluation trials per record
- Frozen comparisons: reconciled MLR, MLR+IFA, and Static Robot-IGFA results from this repository
- Hardware: one NVIDIA GeForce RTX 3090

## Method

Temporal Robot-IGFA V1 changes only the temporal granularity of the frozen Static Robot-IGFA gate. Each trajectory sample is assigned to one of five normalized progress bins: `[0,.2)`, `[.2,.4)`, `[.4,.6)`, `[.6,.8)`, or `[.8,1]`. For each bin containing both current-task and replay samples, all old tasks in that bin are aggregated into one replay anchor. A policy-head projection is applied only when the local current/replay dot product is negative.

The gate uses `alpha=1.0`. It first computes the ordinary full-batch MLR gradient, then adds only

`(n_new_bin / full_batch_size) * (gated_new_bin - original_new_bin)`

to the policy-head gradient. The temporal transformer retains the original MLR gradient. IFA is disabled. Bins without both sides and unresolved samples remain in the baseline loss and receive no gate correction. Stage 6 has no replay and therefore reduces to MLR.

Normalized progress is not a semantic reach/grasp/place phase label. This experiment tests same-progress local functional conditioning only; it does not identify real behavior phases.

## Metadata alignment

The sidecar reuses the existing exact action-window alignment between saved feature records and raw robomimic sequences. Formal feature caches were neither regenerated nor modified.

| Seed | Task 6 | Task 7 | Task 8 | Task 9 | Ambiguous | Unmatched |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1558/1562 (99.744%) | 1480/1480 | 1560/1560 | 1319/1319 | 4 | 0 |
| 43 | 1558/1562 (99.744%) | 1480/1480 | 1560/1560 | 1319/1319 | 4 | 0 |
| 44 | 1558/1562 (99.744%) | 1480/1480 | 1560/1560 | 1319/1319 | 4 | 0 |

The four ambiguous task-6 records per seed have bin `-1`: they remain in `loss_total` but never enter the temporal gate. Missing sidecar keys raise an error; no fallback bin is used.

## Gate=0 parity

A real mixed batch from the formal seed-42 data and stage checkpoints contained six current task-7 and four replay task-6 samples across three candidate bins. With gate strength zero:

| Gradient group | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|
| Policy head | 0 | 0 |
| Transformer | 0 | 0 |

This passes the required `5e-6` maximum-absolute and `1e-6` relative-L2 tolerances.

## Focused tests

Seven focused tests passed. They cover canonical IDs for current/replay filenames, optional sidecar injection, gate-zero parity, same-bin isolation, no-anchor baseline behavior, unresolved-sample baseline behavior, and transformer-gradient invariance.

## Smoke

The seed-42 tasks 6–7, two-epoch smoke used the formal feature cache and an isolated run/buffer. Sidecar construction passed; forward/backward, gate-stat writing, and checkpoint writing completed without NaN/Inf. Stage 6 had zero candidates. Stage 7 recorded 834 candidate bin pairs, 316 conflicts, fire rate 0.378897, and 426/454 batches with a same-bin anchor. Zero-anchor failures were 0. The run wrote 20 per-bin rows, 4 overall rows, and 2 checkpoints. Smoke-only artifacts were removed.

## Formal completeness

| Seed | Checkpoints | Per-bin statistic rows | Overall statistic rows | Evaluation records |
|---:|---:|---:|---:|---:|
| 42 | 24 | 2000 | 400 | 30/30 |
| 43 | 24 | 2000 | 400 | 30/30 |
| 44 | 24 | 2000 | 400 | 30/30 |

Each seed completed tasks 6–9 for 100 epochs. Evaluation used 20 trials, candidate epochs 70/75/80/85/90/95, current-task checkpoint selection, and old-task evaluation at the selected stage checkpoint. The release's stochastic GMM evaluation behavior was preserved.

## Per-seed metrics

All values are percentage points.

| Seed | FWT | NBT | AUC | Selected epochs T6/T7/T8/T9 |
|---:|---:|---:|---:|---|
| 42 | 80.000 | 7.222 | 76.875 | 85 / 70 / 75 / 80 |
| 43 | 78.750 | -6.944 | 82.708 | 95 / 75 / 70 / 75 |
| 44 | 83.750 | 16.944 | 75.833 | 90 / 85 / 90 / 90 |

## Three-seed comparison

Values are mean ± sample standard deviation.

| Method | FWT | NBT | AUC |
|---|---:|---:|---:|
| Frozen MLR | 77.500 ± 1.250 | 2.778 ± 9.279 | 76.284 ± 5.422 |
| Frozen MLR+IFA | 78.333 ± 0.722 | 6.852 ± 8.608 | 74.861 ± 3.898 |
| Static Robot-IGFA | 76.250 ± 5.000 | 1.296 ± 6.587 | 76.285 ± 4.752 |
| Temporal Robot-IGFA V1 | **80.833 ± 2.602** | 5.741 ± 12.013 | **78.472 ± 3.705** |

## Temporal versus Static Robot-IGFA

| Seed | ΔFWT | ΔNBT | ΔAUC |
|---:|---:|---:|---:|
| 42 | -1.250 | 7.778 | -4.896 |
| 43 | 2.500 | -15.556 | 9.063 |
| 44 | 12.500 | 21.111 | 2.396 |
| Mean | **4.583** | **4.444** | **2.188** |

Temporal wins AUC against Static in 2/3 seeds. It meets the prespecified AUC and FWT thresholds, but fails the `mean ΔNBT <= +1.0 pp` threshold. Accordingly, the temporal-versus-static verdict is inconclusive rather than supported.

## Temporal versus frozen MLR

| Seed | ΔFWT | ΔNBT | ΔAUC |
|---:|---:|---:|---:|
| 42 | 3.750 | -3.889 | 6.146 |
| 43 | 1.250 | 0.278 | 1.146 |
| 44 | 5.000 | 12.500 | -0.729 |
| Mean | **3.333** | **2.963** | **2.188** |

Temporal wins AUC against MLR in 2/3 seeds and FWT in 3/3. It satisfies the prespecified practical-benefit conditions: mean AUC gain at least 1 pp, AUC wins in at least 2/3 seeds, and mean FWT change at least -1 pp.

## Per-task FWT and forgetting

| Task | Seed-42 FWT | Seed-43 FWT | Seed-44 FWT | Mean FWT | Mean later-stage forgetting |
|---:|---:|---:|---:|---:|---:|
| 6 | 45 | 20 | 55 | 40.000 | -11.111 |
| 7 | 95 | 100 | 90 | 95.000 | 21.667 |
| 8 | 95 | 100 | 95 | 96.667 | 6.667 |
| 9 | 85 | 95 | 95 | 91.667 | n/a |

Task 7 is the largest source of positive mean forgetting, driven primarily by seed 44 (47.5 pp). Task 6 shows negative mean forgetting because later-stage performance is, on average, better than its low diagonal success. Full stage/task values are retained in `per_stage_success.csv`, `per_task_fwt.csv`, and `per_task_forgetting.csv`.

## Per-bin gate mechanism

The following aggregates weight each bin by candidate/conflict counts across all seeds and lifelong stages.

| Progress bin | Candidate pairs | Conflict pairs | Fire rate | Mean conflict cosine | Projection cost |
|---:|---:|---:|---:|---:|---:|
| 0 | 103495 | 46349 | 0.447838 | -0.193193 | 0.086506 |
| 1 | 103434 | 48774 | 0.471547 | -0.098854 | 0.046616 |
| 2 | 105743 | 44059 | 0.416661 | -0.235585 | 0.098104 |
| 3 | 106044 | 43660 | 0.411716 | -0.201419 | 0.082986 |
| 4 | 104185 | 45502 | 0.436742 | -0.114558 | 0.049963 |
| Overall | 522901 | 228344 | 0.436687 | -0.167125 | 0.072981 |

Across all batches, including replay-free stage 6, same-bin candidate coverage is 0.818306 (258994/316500). For stages 7–9 alone, per-seed/stage batch coverage is approximately 0.948–0.973. No candidate computation reported a zero replay-anchor norm (`zero_anchor_count=0`). Bin 2 has the most negative conflict cosine and largest projection cost, while bin 1 fires most frequently but has weaker conflicts.

## Seed consistency

Temporal FWT standard deviation (2.602) is lower than Static (5.000), and Temporal AUC standard deviation (3.705) is lower than Static (4.752). NBT remains highly variable (12.013), and same-seed directions are not uniform: seed 42 loses AUC to Static, while seed 44 loses AUC to MLR. With `n=3`, these are descriptive observations, not significance claims.

## Observed

- The same-bin gate is active: 81.831% overall candidate coverage and 43.669% candidate-pair fire rate.
- Temporal improves mean FWT and AUC over both Static and frozen MLR.
- Temporal beats Static AUC in seeds 43 and 44, but loses by 4.896 pp in seed 42.
- Retention behavior is unstable: NBT changes range from -15.556 to +21.111 pp versus Static.
- All five bins contain substantial conflict; temporal heterogeneity does not disappear after coarse progress conditioning.

## Inferred

- Coarse same-progress conditioning is more promising than whole-batch aggregation for average FWT/AUC, but this three-seed result does not establish a stable retention advantage over Static.
- Aggregating all old tasks within one progress bin may still mix opposing task-specific relations. The substantial within-bin conflict density makes per-task temporal anchors a motivated next diagnostic, but does not by itself prove aggregate cancellation.

## Not established

- The five normalized bins do not establish semantic reach, grasp, transport, or place phases.
- No statistical significance is claimed from three seeds.
- A causal explanation for seed-42 AUC loss or seed-44 task-7 forgetting is not established.
- A per-task temporal gate, learned phase encoder, behavior clustering, transformer projection, IFA combination, or alpha sweep has not been evaluated.

## Verdict

`TEMPORAL_ROBOT_IGFA_INCONCLUSIVE`

Practical benefit versus frozen MLR: `SUPPORTED`.

H5—same-progress local replay conditioning is superior to whole-batch Static conditioning—is not fully supported under the prespecified rule because the mean NBT delta versus Static is +4.444 pp, despite a +2.188 pp mean AUC gain, +4.583 pp mean FWT gain, and 2/3 AUC wins.

## Exact successful commands

```bash
cd /data/lifelong_mlr_ifa_minrep
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/conda_envs/mlr_ifa_minrep
source scripts/_common.sh

pytest -q tests/test_temporal_robot_igfa.py
SEED=42 bash scripts/run_object_mlr_temporal_robot_igfa_smoke.sh

SEED=42 bash scripts/run_object_mlr_temporal_robot_igfa.sh
SEED=42 bash test.sbatch mlr_temporal_igfa
SEED=43 bash scripts/run_object_mlr_temporal_robot_igfa.sh
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 SEED=43 bash test.sbatch mlr_temporal_igfa
SEED=44 bash scripts/run_object_mlr_temporal_robot_igfa.sh
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 SEED=44 bash test.sbatch mlr_temporal_igfa

python scripts/compute_temporal_robot_igfa_metrics.py
```

The offline flags force use of the already-populated frozen CLIP cache and do not change model weights or evaluation behavior. Successful evaluation records were retained when completing interrupted evaluation batches; no completed record was rerun or selected by outcome.

## Next action

Diagnose per-task relations within each progress bin, especially bin 2 and seed-44 task-7 retention, before implementing a per-task + temporal anchor. Do not run an alpha sweep first.
