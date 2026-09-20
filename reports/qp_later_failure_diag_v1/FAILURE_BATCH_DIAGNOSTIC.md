# Failure batch localization and capture

The latest event file is `/root/qp_later_failure_diag_v1/run/events/task8_epoch011_attempt006.jsonl`. It contains 185 valid JSON events, covering Python batch indices 0 through 184. Batch 184 is the last successful event at global/optimizer/scheduler step 41984/3584/3584. By the frozen rule, the actual failure is Python batch 185. The periodic STATUS `batch=180` is a successful-step counter refreshed every 20 steps, not the failed Python batch index.

Restored the saved pre-index-180 state and replayed batches 180–184 in sequence. All five event comparisons matched, including record indices, source composition, loss, LR, trigger/constraint/active counts, KKT, projection residual, norms, correction ratio, and optimizer/scheduler/global steps. Maximum observed absolute difference was 0.

A reload/restore comparison of `PRE_FAILURE_STATE.pt` passed. The frozen batch 185 input fingerprints and runtime hashes were written. The runtime function identity check passed (`core.solve is solver.solve`). One update on batch 185 raised the expected frozen-threshold assertion at KKT `2.3018110750783194e-8`, exactly matching the historical value.

Before extended polish: KKT `4.864608038612772e-08`; actual polish counts: base 10000, continuation 20000, extended 100000 (total 130000). After 100k extra sweeps KKT remained `2.3018110750783194e-08`. The full eight-point curve is in `t8_e11_b180/convergence.csv`; before/after QP arrays are in the corresponding NPZ files. Solver comparison has not been run.

`formal_training_resumed = false`. The failed live `update` had already incremented AdamW and scheduler once before the solver assertion, so that post-failure in-memory state is not a valid continuation checkpoint. All subsequent work must restore `PRE_FAILURE_STATE.pt` or operate offline on the saved QP.
