# Seed41 Coreset + IFA + zero-response residual

Status: scientific/engineering audits passed; process exit must be verified separately.

| Stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 75% | — | — | — |
| 7 | 80% | 65% | — | — |
| 8 | 65% | 70% | 100% | — |
| 9 | 35% | 55% | 100% | 20% |

| Metric | Coreset baseline % | New % | Difference pp |
|---|---:|---:|---:|
| AUC | 70.10416667 | 61.77083333 | -8.33333333 |
| FWT | 67.50000000 | 65.00000000 | -2.50000000 |
| NBT | -5.55555556 | 5.83333333 | +11.38888889 |
| OldFinal | 71.66666667 | 63.33333333 | -8.33333333 |
| FinalAll | 66.25000000 | 52.50000000 | -13.75000000 |

Prespecified progression gate: {'AUC': False, 'FWT': False, 'NBT': False, 'OldFinal': False}

Limitations: one exploratory seed; unmatched total parameter count; original global action RNG is not isolated per trajectory; pretrained state metadata unknown; parent schedule engineering completion retained; shared network can forget; zero direct residual response does not imply whole-policy invariance. No automatic extension.

Counts: 400 epochs, 105500 optimizer steps, 4 final checkpoints, 200 formal episodes, 20 vector attempts. Integration: 4 optimizer calls, zero environment steps. Baseline was not retrained or re-evaluated.
