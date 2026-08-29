# Optimizer-Aware Trajectory Signal Stability Diagnosis

## Identity

- Verdict: `OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_INCONCLUSIVE`
- Branch: `research/optimizer-aware-trajectory-signal`
- Starting SHA: `9795a284a835da0d20c63e22f1bac4bc45113493`
- Final SHA: branch HEAD containing this report (reported in the external handoff)

## Input

- Anchors: 267
- Raw H100 cases: 261
- Raw H300 cases: 249
- Source SHA: `9795a284a835da0d20c63e22f1bac4bc45113493`

## History completeness

- H100 complete: 249; per seed: {42: 83, 43: 83, 44: 83}
- H300 complete: 237; per seed: {42: 79, 43: 79, 44: 79}
- H12 evidence floor: met

## Seed-stage table

| Seed | Stage | Single H100 AUC | EMA3 H100 AUC | Single H300 AUC | EMA3 H300 AUC |
|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 0.5971 | 0.6406 | 0.6722 | 0.6789 |
| 42 | 9 | 0.6911 | 0.6667 | 0.7386 | 0.7869 |
| 43 | 8 | 0.6261 | 0.6087 | 0.7257 | 0.7083 |
| 43 | 9 | 0.6050 | 0.6029 | 0.6355 | 0.6010 |
| 44 | 8 | 0.6532 | 0.6296 | 0.5942 | 0.5519 |
| 44 | 9 | 0.7024 | 0.7328 | 0.7199 | 0.7407 |

## H100

| Feature | AUC | AP | LOSO BA | LOSO false-safe |
|---|---:|---:|---:|---:|
| single_raw | 0.6411 | 0.4754 | 64.96% | 13.65% |
| ema3_raw | 0.6340 | 0.4812 | 67.19% | 13.65% |
| mean2_raw | 0.6410 | 0.4962 | NA | NA |
| mean3_raw | 0.6242 | 0.4664 | NA | NA |
| single_relative | 0.6417 | 0.4763 | 64.96% | 13.65% |
| ema3_relative | 0.6354 | 0.4810 | 67.19% | 13.65% |

EMA3 AUC gain: -0.0071; LOSO BA gain: +2.23 pp.

## H300

| Feature | AUC | AP | LOSO BA | LOSO false-safe |
|---|---:|---:|---:|---:|
| single_raw | 0.6772 | 0.5384 | 66.19% | 9.28% |
| ema3_raw | 0.6750 | 0.5128 | 67.23% | 12.24% |
| mean2_raw | 0.6895 | 0.4970 | NA | NA |
| mean3_raw | 0.6618 | 0.4836 | NA | NA |
| single_relative | 0.6782 | 0.5389 | 66.21% | 10.13% |
| ema3_relative | 0.6757 | 0.5117 | 67.23% | 12.24% |

Single/EMA3 AUC: 0.6772/0.6750; single/EMA3 LOSO BA: 66.19%/67.23%.

## Horizon transition

- Paired cases: 237
- safe100 -> harmful300: 21
- Delayed harmful rate among H100-safe: 14.00%
- harmful100 -> safe300: 28
- Delayed trajectory finding: not established

## Scale

| Feature | AUC | AP | Delta AUC vs raw |
|---|---:|---:|---:|
| actual_step_score | 0.6411 | 0.4754 | +0.0000 |
| actual_step_cosine | 0.6943 | 0.5427 | +0.0532 |
| actual_step_relative | 0.6417 | 0.4763 | +0.0006 |
| delta_norm | 0.4418 | 0.3557 | -0.1993 |
| old_grad_norm | 0.4707 | 0.3587 | -0.1704 |
| norm_product | 0.4607 | 0.3593 | -0.1804 |

Inference: scale normalization improves ranking.

## Heterogeneity

`SEED_STAGE_HETEROGENEITY_MODERATE`; valid six-group EMA3 H100 AUC range = 0.1299.

## Observed

The fixed EMA3, auxiliary trajectory rankings, LOSO folds, scale controls, and H100/H300 transitions are direct computations from the committed anchor and future-case CSV files.

## Inferred

H12 is `OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_INCONCLUSIVE`. The fixed recent-history aggregation does not meet the preregistered support rule.

## Not established

Anchor spacing is 100 updates. Trajectory features come from targeted continuation, not the exact original training trajectory. H100/H300 labels are BC-loss retention, not rollout success. LOSO across three seeds is descriptive. EMA3 is a diagnostic predictor, not a validated training method.

## Next action

Resolve cross-seed and horizon instability before any trajectory intervention.
