# Optimizer-Step Direction and Horizon Reversal Diagnosis

## Identity

- Verdict: `OPTIMIZER_STEP_DIRECTIONAL_RISK_SUPPORTED`
- Flags: `HORIZON_REVERSAL_MATERIAL`, `REVERSAL_EXPLAINS_DIRECTIONAL_DEGRADATION`, `PERSISTENT_HARMFUL_DIRECTIONAL_SIGNAL`, `DIRECTIONAL_NORMALIZATION_ADVANTAGE`
- Branch: `research/optimizer-step-direction-horizon-reversal`
- Starting SHA: `9691aca9dbebc4481aaa92fba192b6ff5d2b1b84`
- Final SHA: branch HEAD containing this report (reported in the external handoff)

## Inputs

- Anchors: 267
- H100: 261
- H300: 249
- Paired: 249

## Primary H100

| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |
|---|---:|---:|---:|---:|---:|
| raw | 0.6587 | 0.4733 | 66.14% | 63.04% | 13.03% |
| cosine | 0.7153 | 0.5425 | 66.73% | 63.04% | 13.03% |
| relative | 0.6593 | 0.4743 | 66.14% | 63.04% | 13.03% |

Cosine gain versus raw: AUC +0.0567; LOSO BA +0.59 pp.

## H300

| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |
|---|---:|---:|---:|---:|---:|
| raw | 0.6934 | 0.5367 | 68.63% | 77.50% | 7.23% |
| cosine | 0.7393 | 0.5550 | 69.19% | 65.00% | 11.24% |
| relative | 0.6944 | 0.5373 | 68.57% | 75.00% | 8.03% |

## LOSO cosine folds

| Seed | tau | BA | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|
| 42 | 0.0022885 | 67.81% | 56.67% | 78.95% |
| 43 | 0.00414567 | 65.40% | 56.25% | 74.55% |
| 44 | -0.00468242 | 67.28% | 76.67% | 57.89% |

## Seed-stage

| Seed | Stage | H100 cosine AUC | H300 cosine AUC |
|---:|---:|---:|---:|
| 42 | 8 | 0.6693 | 0.7138 |
| 42 | 9 | 0.7646 | 0.7807 |
| 43 | 8 | 0.7413 | 0.8301 |
| 43 | 9 | 0.7098 | 0.7143 |
| 44 | 8 | 0.7053 | 0.6458 |
| 44 | 9 | 0.7406 | 0.7478 |

Direction heterogeneity: `DIRECTION_HETEROGENEITY_MODERATE`; six-group H100 cosine AUC range = 0.0952.

## Horizon transition

SS=141, SH=21, HH=59, HS=28; flip rate 19.68%; delayed rate among H100-safe 12.96%; recovery rate among H100-harmful 32.18%.

## Stable-label ranking

Cases/positives 200/59; cosine AUC/AP 0.8031/0.5878; LOSO BA 75.94%; valid folds 3/3.

## Delayed harmful

Cases/positives 162/21; cosine AUC/AP 0.6039/0.1742; LOSO BA 47.77%; valid folds 3/3.

## Persistent harmful

Cases/positives 87/59; cosine AUC/AP 0.7446/0.8670; LOSO BA 66.31%; valid folds 3/3.

## Scale controls

| Feature | AUC | AP |
|---|---:|---:|
| actual_step_score | 0.6587 | 0.4733 |
| actual_step_cosine | 0.7153 | 0.5425 |
| actual_step_relative | 0.6593 | 0.4743 |
| delta_norm | 0.4815 | 0.3557 |
| old_grad_norm | 0.4492 | 0.3247 |
| norm_product | 0.4691 | 0.3507 |

## Observed

The directional rankings, cross-seed thresholds, transition counts, stable/delayed/persistent subgroup rankings, and norm controls are direct computations from the committed anchor and raw H100/H300 CSVs.

## Inferred

H13 is `OPTIMIZER_STEP_DIRECTIONAL_RISK_SUPPORTED`. Direction normalization meets the preregistered primary support rule. Horizon interpretations are limited to the listed descriptive flags.

## Not established

Actual-step cosine is a post-update diagnostic score, not a pre-update deployable gate. The targeted continuations are distribution-matched, not the exact historical lifelong trajectory. H100/H300 labels are old-task BC-loss drift, not rollout-success forgetting. Three-seed LOSO is descriptive, not a statistical significance claim. Horizon reversal does not by itself prove optimizer path dependence as the unique cause.

## Next action

Run Candidate AdamW-Step Cosine Feasibility Diagnosis before any guard training.
