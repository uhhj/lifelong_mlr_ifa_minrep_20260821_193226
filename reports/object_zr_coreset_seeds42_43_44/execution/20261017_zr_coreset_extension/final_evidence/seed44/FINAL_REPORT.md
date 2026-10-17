# Seed44 Coreset + IFA + zero-response residual

Status: scientific/engineering audits passed; process exit must be verified separately.

| Stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 50% | — | — | — |
| 7 | 60% | 95% | — | — |
| 8 | 70% | 100% | 100% | — |
| 9 | 65% | 100% | 100% | 95% |

| Metric | Coreset baseline % | New % | Difference pp |
|---|---:|---:|---:|
| AUC | 86.25000000 | 88.64583333 | +2.39583333 |
| FWT | 86.25000000 | 85.00000000 | -1.25000000 |
| NBT | 0.00000000 | -6.66666667 | -6.66666667 |
| OldFinal | 80.00000000 | 88.33333333 | +8.33333333 |
| FinalAll | 83.75000000 | 90.00000000 | +6.25000000 |

Prespecified progression gate: {'AUC': True, 'FWT': False, 'NBT': True, 'OldFinal': True}

Limitations: one exploratory seed; unmatched total parameter count; original global action RNG is not isolated per trajectory; pretrained state metadata unknown; parent schedule engineering completion retained; shared network can forget; zero direct residual response does not imply whole-policy invariance. No automatic extension.

Counts: 400 epochs, 105500 optimizer steps, 4 final checkpoints, 200 formal episodes, 20 vector attempts. Integration: 4 optimizer calls, zero environment steps. Baseline was not retrained or re-evaluated.
