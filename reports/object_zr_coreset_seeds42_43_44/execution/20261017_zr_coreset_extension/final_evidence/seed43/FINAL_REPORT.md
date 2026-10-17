# Seed43 Coreset + IFA + zero-response residual

Status: scientific/engineering audits passed; process exit must be verified separately.

| Stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 15% | — | — | — |
| 7 | 75% | 100% | — | — |
| 8 | 20% | 100% | 100% | — |
| 9 | 25% | 100% | 100% | 60% |

| Metric | Coreset baseline % | New % | Difference pp |
|---|---:|---:|---:|
| AUC | 94.37500000 | 73.43750000 | -20.93750000 |
| FWT | 88.75000000 | 68.75000000 | -20.00000000 |
| NBT | -10.00000000 | -8.33333333 | +1.66666667 |
| OldFinal | 93.33333333 | 75.00000000 | -18.33333333 |
| FinalAll | 95.00000000 | 71.25000000 | -23.75000000 |

Prespecified progression gate: {'AUC': False, 'FWT': False, 'NBT': False, 'OldFinal': False}

Limitations: one exploratory seed; unmatched total parameter count; original global action RNG is not isolated per trajectory; pretrained state metadata unknown; parent schedule engineering completion retained; shared network can forget; zero direct residual response does not imply whole-policy invariance. No automatic extension.

Counts: 400 epochs, 105500 optimizer steps, 4 final checkpoints, 200 formal episodes, 20 vector attempts. Integration: 4 optimizer calls, zero environment steps. Baseline was not retrained or re-evaluated.
