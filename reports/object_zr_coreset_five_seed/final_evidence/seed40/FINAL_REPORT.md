# Seed40 Coreset + IFA + zero-response residual

Status: scientific/engineering audits passed; process exit must be verified separately.

| Stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 20% | — | — | — |
| 7 | 60% | 60% | — | — |
| 8 | 25% | 95% | 100% | — |
| 9 | 25% | 100% | 100% | 40% |

| Metric | Coreset baseline % | New % | Difference pp |
|---|---:|---:|---:|
| AUC | 81.14583333 | 64.37500000 | -16.77083333 |
| FWT | 75.00000000 | 55.00000000 | -20.00000000 |
| NBT | -10.83333333 | -18.05555556 | -7.22222222 |
| OldFinal | 71.66666667 | 75.00000000 | +3.33333333 |
| FinalAll | 77.50000000 | 66.25000000 | -11.25000000 |

Prespecified progression gate: {'AUC': False, 'FWT': False, 'NBT': True, 'OldFinal': True}

Limitations: one exploratory seed; unmatched total parameter count; original global action RNG is not isolated per trajectory; pretrained state metadata unknown; parent schedule engineering completion retained; shared network can forget; zero direct residual response does not imply whole-policy invariance. No automatic extension.

Counts: 400 epochs, 105500 optimizer steps, 4 final checkpoints, 200 formal episodes, 20 vector attempts. Integration: 4 optimizer calls, zero environment steps. Baseline was not retrained or re-evaluated.
