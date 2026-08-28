# LR / scheduler migration follow-up

## Result

Classification: **B** — the release schedule begins at exactly zero and then becomes non-zero under its long 500-step warmup. It is not permanently stuck at zero.

The active path is `Sequential.start_task()` → `make_linear_schedule_with_warmup()` → `LambdaLR`; `MLR.learn_all_tasks()` calls `scheduler.step()` once after each epoch when `epoch > 0`. The formal seed-42 task-6 loader has 157 batches, so `total_steps=100×157=15700`, even though the scheduler itself advances per epoch rather than per optimizer update.

## Exact trace

Base AdamW LR before scheduler construction: `0.0001`.

| Point | LR |
|---|---:|
| after `LambdaLR` construction | 0 |
| scheduler step 1 | 2e-7 |
| scheduler step 2 | 4e-7 |
| scheduler step 10 | 2e-6 |
| scheduler step 100 | 2e-5 |
| scheduler step 1000 (pure schedule trace) | 9.67105263158e-5 |
| scheduler step 5000 (pure schedule trace) | 7.03947368421e-5 |
| scheduler step 10000 (pure schedule trace) | 3.75e-5 |
| end of actual smoke epoch 0 | 0 |
| end of actual smoke epoch 1 | 2e-7 |

The 1000/5000/10000 values are pure scheduler-step probes and are not reached by the current 100-epoch loop, which advances the scheduler only once per epoch.

## Interpretation

The historical log's `lr: 0` was numerically real at initialization, not merely display rounding. Later optimizer updates use non-zero but initially very small learning rates. This is existing release behavior and is unchanged for the offline diagnosis.

Under the prespecified rule, this is **not** a blocker for the next formal training because it is not case C. Before another 100-epoch intervention experiment, the epoch-versus-update scheduler semantics must nevertheless be frozen explicitly and applied identically to Temporal V1 and its intervention comparator.
