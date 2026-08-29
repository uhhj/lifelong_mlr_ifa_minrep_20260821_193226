# Retention Score Ranking and Calibration Diagnosis

## Verdict

`LOCAL_RETENTION_SCORE_CALIBRATION_SUPPORTED`

## Attribution

`FIRST_ORDER_MAGNITUDE_SUFFICIENT`

## Identity

- Branch: `research/retention-score-ranking-calibration`
- Starting SHA: `ff710880fc8c1fee68a4ff44aa63c4d1711e5aa2`
- Final SHA: branch HEAD at handoff

## Input

- Source: `reports/finite_step_functional_probe_validity/finite_step_probe_raw.csv`
- Rows: 720
- Unique short cases: 180
- V1/V2 cases: 90/90

## Primary

Method V1, finite-step rho=0.10, short transitions, leave-one-seed-out threshold calibration.
Endpoint-harmful prevalence is 56.67%; AP is 0.8633, or +0.2967 above prevalence.
LOSO all-case false-safe is 13.33%; harmful-conditional miss rate is 23.53%. These denominators are not interchangeable.

## Ranking and calibration

| Method | Score | rho | ROC-AUC | AP | Zero BA | LOSO BA | Zero FS | LOSO FS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | first_order | NA | 0.7954 | 0.8461 | 50.00% | 71.87% | 56.67% | 14.44% |
| V1 | finite_step | 0.05 | 0.8059 | 0.8574 | 50.00% | 73.15% | 56.67% | 14.44% |
| V1 | finite_step | 0.10 | 0.8125 | 0.8633 | 50.00% | 75.41% | 56.67% | 13.33% |
| V1 | finite_step | 0.25 | 0.8396 | 0.8843 | 50.00% | 78.36% | 56.67% | 10.00% |
| V1 | finite_step | 0.50 | 0.8894 | 0.9252 | 50.00% | 83.71% | 56.67% | 15.56% |
| V2 | first_order | NA | 0.7975 | 0.8359 | 50.00% | 71.11% | 50.00% | 20.00% |
| V2 | finite_step | 0.10 | 0.8227 | 0.8562 | 50.00% | 73.33% | 50.00% | 17.78% |

## Horizon ranking curve

| rho | ROC-AUC | AP | LOSO BA | LOSO FS |
|---:|---:|---:|---:|---:|
| 0.05 | 0.8059 | 0.8574 | 73.15% | 14.44% |
| 0.10 | 0.8125 | 0.8633 | 75.41% | 13.33% |
| 0.25 | 0.8396 | 0.8843 | 78.36% | 10.00% |
| 0.50 | 0.8894 | 0.9252 | 83.71% | 15.56% |

## Calibration held-out folds

| Held-out seed | Threshold learned from | tau | Test BA | Sensitivity | Specificity | All-case false-safe | Harmful miss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | 43+44 | -0.542544 | 64.03% | 58.82% | 69.23% | 23.33% | 41.18% |
| 43 | 42+44 | -0.511654 | 79.64% | 82.35% | 76.92% | 10.00% | 17.65% |
| 44 | 42+43 | -0.511654 | 82.58% | 88.24% | 76.92% | 6.67% | 11.76% |

## First-order vs finite

- ROC-AUC: 0.7954 vs 0.8125 (delta +0.0171).
- AP: 0.8461 vs 0.8633 (delta +0.0173).
- LOSO BA: 71.87% vs 75.41% (+3.54 pp).
- LOSO all-case false-safe: 14.44% vs 13.33% (finite reduction +1.11 pp).

## Distribution

- Harmful V1 rho0.10 scores: n=51, median=-0.453505, q25/q75=-0.523761/-0.373452.
- Safe V1 rho0.10 scores: n=39, median=-0.603424, q25/q75=-0.669862/-0.532710.

## Secondary

- V2 first-order ROC-AUC/LOSO BA: 0.7975/71.11%.
- V2 finite rho0.10 ROC-AUC/LOSO BA: 0.8227/73.33%.
- Material-subset ranking direction is consistent with the full V1 rho0.10 set (material ROC-AUC 0.8312).
- Stage 7: ROC-AUC 0.8409, AP 0.9488, prevalence 73.33%.
- Stage 8: ROC-AUC 0.7778, AP 0.8045, prevalence 50.00%.
- Stage 9: ROC-AUC 0.8600, AP 0.8848, prevalence 55.56%.

## Observed

ROC-AUC, AP, zero-threshold classification, LOSO fold thresholds, held-out confusion metrics, distributions, and secondary slices are direct deterministic calculations from the committed CSV.

## Inferred

H10 is `LOCAL_RETENTION_SCORE_CALIBRATION_SUPPORTED` and the preregistered attribution is `FIRST_ORDER_MAGNITUDE_SUFFICIENT`. Oracle pooled thresholds are recorded only as descriptive upper bounds in `threshold_summary.csv` and are not verdict evidence.

## Not established

LOSO across three seeds is a small-sample descriptive generalization test, not a statistical significance claim. A calibrated offline threshold does not establish stability during online training. The finite-step score uses the future checkpoint chord and is not deployable online. This diagnosis concerns old BC-loss drift, not rollout-success retention.

## Verdict

`LOCAL_RETENTION_SCORE_CALIBRATION_SUPPORTED`

## Next action

Test a margin-aware calibrated first-order retention gate on Temporal V1.
