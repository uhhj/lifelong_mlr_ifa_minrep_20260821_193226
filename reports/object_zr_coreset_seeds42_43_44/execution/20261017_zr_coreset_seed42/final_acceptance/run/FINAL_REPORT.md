# Seed42 Coreset + IFA + zero-response residual

Status: scientific/engineering audits passed; process exit must be verified separately.

| Stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 35% | — | — | — |
| 7 | 65% | 75% | — | — |
| 8 | 45% | 100% | 100% | — |
| 9 | 55% | 100% | 100% | 75% |

| Metric | Coreset baseline % | New % | Difference pp |
|---|---:|---:|---:|
| AUC | 68.12500000 | 79.16666667 | +11.04166667 |
| FWT | 65.00000000 | 71.25000000 | +6.25000000 |
| NBT | -5.83333333 | -15.00000000 | -9.16666667 |
| OldFinal | 73.33333333 | 85.00000000 | +11.66666667 |
| FinalAll | 66.25000000 | 82.50000000 | +16.25000000 |

Prespecified progression gate: {'AUC': True, 'FWT': True, 'NBT': True, 'OldFinal': True}

Limitations: one exploratory seed; unmatched total parameter count; original global action RNG is not isolated per trajectory; pretrained state metadata unknown; parent schedule engineering completion retained; shared network can forget; zero direct residual response does not imply whole-policy invariance. No automatic extension.

Counts: 400 epochs, 105500 optimizer steps, 4 final checkpoints, 200 formal episodes, 20 vector attempts. Integration: 4 optimizer calls, zero environment steps. Baseline was not retrained or re-evaluated.

Independent exit acceptance PASS: original pre-update audit failure exits=1 retained; external metadata normalization recovery exit=0. Model/loss/data code and original contract unchanged.
