# Fixed-kappa confirmation v2

FIXED_KAPPA_CONFIRMATION_INCONCLUSIVE

## Execution integrity

Six fixed runs and six fixed-ep99 20-trial-per-cell success matrices completed. No best-epoch selection or performance stopping. The paired task6 gates passed before task7 for every seed.

Implementation commit: 3284c4213c7a7694ee906aed5505aa06bb1f3c44. Full source hashes and precheck provenance are in IMPLEMENTATION.json and PREFLIGHT_COMPLETE.json.

Input/config/cache hashes: FROZEN_INPUTS.json. Logging and snapshot parity: GPU_PARITY.json. Per-seed task6 checks: TASK6_PARITY_SEED*.json. Raw matrices/logs/events/snapshots remain in runs/fixed_kappa_confirmation_v2/.

## Primary confirmation: new seeds 45/46/47

| method | FWT | NBT | AUC | final success |
| --- | --- | --- | --- | --- |
| mlr | 64.58 +/- 9.46 | -13.33 +/- 4.17 | 71.94 +/- 12.61 | 70.83 +/- 8.51 |
| soft | 62.50 +/- 5.45 | -19.07 +/- 14.73 | 72.81 +/- 4.38 | 75.42 +/- 4.73 |

| seed | delta FWT | delta NBT | delta AUC | delta final |
| --- | --- | --- | --- | --- |
| 45 | +12.5000 | +5.0000 | +10.4167 | +8.7500 |
| 46 | -15.0000 | -20.2778 | -5.0000 | +3.7500 |
| 47 | -3.7500 | -1.9444 | -2.8125 | +1.2500 |

All differences are soft minus MLR in percentage points. Lower NBT alone is not evidence of a better stability/plasticity tradeoff. The decision uses only new seeds and the frozen descriptive rule.

## Historical replication context: 42/43/44

| method | FWT | NBT | AUC | final success |
| --- | --- | --- | --- | --- |
| mlr | 82.50 +/- 6.50 | 3.89 +/- 5.30 | 80.10 +/- 8.01 | 78.33 +/- 13.83 |
| soft | 82.50 +/- 5.00 | 0.74 +/- 3.61 | 81.70 +/- 6.16 | 80.00 +/- 5.73 |

| seed | delta FWT | delta NBT | delta AUC | delta final |
| --- | --- | --- | --- | --- |
| 42 | +7.5000 | -2.7778 | +9.1667 | +11.2500 |
| 43 | -8.7500 | -6.1111 | -6.2500 | -1.2500 |
| 44 | +1.2500 | -0.5556 | +1.8750 | -5.0000 |

## Exploratory pooled summary: six seeds

| method | FWT | NBT | AUC | final success |
| --- | --- | --- | --- | --- |
| mlr | 73.54 +/- 12.21 | -4.72 +/- 10.35 | 76.02 +/- 10.45 | 74.58 +/- 11.06 |
| soft | 72.50 +/- 11.91 | -9.17 +/- 14.48 | 77.26 +/- 6.82 | 77.71 +/- 5.33 |

Descriptive context only; pooling cannot overturn an unsuccessful new-seed confirmation. No exaggerated significance claims are made.

## Per-task localization

| seed | method | task | first | final | column AUC | peak-to-final |
| --- | --- | --- | --- | --- | --- | --- |
| 45 | mlr | 6 | 25.0000 | 25.0000 | 21.2500 | 0.0000 |
| 45 | mlr | 7 | 85.0000 | 95.0000 | 93.3333 | 5.0000 |
| 45 | mlr | 8 | 70.0000 | 90.0000 | 80.0000 | 0.0000 |
| 45 | mlr | 9 | 35.0000 | 35.0000 | 35.0000 | 0.0000 |
| 45 | soft | 6 | 25.0000 | 50.0000 | 36.2500 | 0.0000 |
| 45 | soft | 7 | 100.0000 | 70.0000 | 85.0000 | 30.0000 |
| 45 | soft | 8 | 80.0000 | 100.0000 | 90.0000 | 0.0000 |
| 45 | soft | 9 | 60.0000 | 60.0000 | 60.0000 | 0.0000 |
| 46 | mlr | 6 | 5.0000 | 30.0000 | 53.7500 | 65.0000 |
| 46 | mlr | 7 | 100.0000 | 95.0000 | 90.0000 | 5.0000 |
| 46 | mlr | 8 | 100.0000 | 90.0000 | 95.0000 | 10.0000 |
| 46 | mlr | 9 | 80.0000 | 80.0000 | 80.0000 | 0.0000 |
| 46 | soft | 6 | 5.0000 | 50.0000 | 48.7500 | 35.0000 |
| 46 | soft | 7 | 25.0000 | 75.0000 | 60.0000 | 5.0000 |
| 46 | soft | 8 | 100.0000 | 90.0000 | 95.0000 | 10.0000 |
| 46 | soft | 9 | 95.0000 | 95.0000 | 95.0000 | 0.0000 |
| 47 | mlr | 6 | 0.0000 | 30.0000 | 37.5000 | 45.0000 |
| 47 | mlr | 7 | 85.0000 | 95.0000 | 90.0000 | 0.0000 |
| 47 | mlr | 8 | 100.0000 | 95.0000 | 97.5000 | 5.0000 |
| 47 | mlr | 9 | 90.0000 | 90.0000 | 90.0000 | 0.0000 |
| 47 | soft | 6 | 0.0000 | 55.0000 | 43.7500 | 10.0000 |
| 47 | soft | 7 | 95.0000 | 95.0000 | 95.0000 | 0.0000 |
| 47 | soft | 8 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| 47 | soft | 9 | 65.0000 | 65.0000 | 65.0000 | 0.0000 |

Worst-old-task final success is included in primary_metrics.csv. Full success matrices are in RESULTS.json.

## Event distribution

| scope | batches | valid reference | triggers | trigger rate | ordinary cancelled | strong z |
| --- | --- | --- | --- | --- | --- | --- |
| global | 316500 | 268021 | 2729 | 1.0182% | 72923 | 2729 |
| phase/00-09 | 31650 | 26801 | 98 | 0.3657% | 8678 | 98 |
| phase/10-29 | 63300 | 53597 | 461 | 0.8601% | 14002 | 461 |
| phase/30-69 | 126600 | 107228 | 1076 | 1.0035% | 26044 | 1076 |
| phase/70-99 | 94950 | 80395 | 1094 | 1.3608% | 24199 | 1094 |
| seed/45 | 105500 | 89362 | 949 | 1.0620% | 24283 | 949 |
| seed/46 | 105500 | 89324 | 823 | 0.9214% | 24219 | 823 |
| seed/47 | 105500 | 89335 | 957 | 1.0712% | 24421 | 957 |
| task/6 | 47100 | 0 | 0 | - | 0 | 0 |
| task/7 | 68100 | 66990 | 1267 | 1.8913% | 16729 | 1267 |
| task/8 | 92700 | 92434 | 1034 | 1.1186% | 23536 | 1034 |
| task/9 | 108600 | 108597 | 428 | 0.3941% | 32658 | 428 |

Continuous distributions, conditional denominators, z quantiles within triggers, cosine, gradient norms and correction ratios are in event_summary.json/CSV. Trigger rate is not the total new/old negative-gradient rate.

## Limitations

Only three new seeds. Kappa .5 was selected from earlier experiments, and adjacent kappas did not establish a stable positive region. The historical front end remains use_state=False. Algebraic event measurements, checkpoint snapshots and local BC diagnostics do not establish closed-loop causal protection. No counterfactual mechanism trials were run as part of this confirmation.

The preregistered thresholds are descriptive research decisions rather than significance tests. All seed/task results are reported regardless of outcome.


## Audited recovery of a launcher failure

The initial queue stopped after seed45 MLR training completed because overlapping PyTorch 1.13 standalone launchers shared a C10d store. Communication was isolated with separate ephemeral endpoints, verified with overlapping real NCCL processes and peer-exit survival. Scientific settings and original source files remained unchanged.

Seed45 soft task6 was replayed in a separate audit directory solely to reconstruct its lost live state. The original 15700-step loss trace, model, AdamW, scheduler, RNG and replay contents matched exactly before the same live process continued task7. Original completed checkpoints/results were retained; reconstruction outputs were not counted as an additional seed or arm.

Both completed seed45 task6 evaluations were reused. Model/RNG/config/backend equality at all remaining native evaluation boundaries was verified without robot trials. The interrupted, incomplete stage7 evaluation was replayed under its original protocol; failed-attempt logs remain archived. The primary matrix retains 20 trials per cell and no outcome-based reruns. Result preservation is audited in recovery_v1/RESULT_PRESERVATION.json.
