# Scheduler semantics

`repro_scheduler_mode` defaults to `legacy_epoch`, preserving the starting baseline.

| Mode | Optimizer update | Scheduler update | Interpretation |
|---|---|---|---|
| `legacy_epoch` | every batch | once after epochs greater than zero | exact release behavior |
| `batch_step` | every batch | once immediately after every optimizer update | matches the units used by warmup=500 and total=`epochs*batches` |
| `constant` | every batch | scheduler absent | diagnostic control at optimizer LR 1e-4 |

The linear lambda and warmup/total-step arithmetic are unchanged. Task6 diagnostics add checkpoints 9,19,…,99 while formal selection remains restricted to 70,75,80,85,90,95. Instrumentation writes one LR row and one true mean-loss row per epoch without changing the objective.
