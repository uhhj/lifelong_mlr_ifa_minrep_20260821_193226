# v4 numerical continuation runtime

Training remains stopped. Standard QP and v3 L-BFGS-B are unchanged. Only a failed first fallback invokes trust-constr from the original 30k solution, using the first KKT-certified iterate.

The frozen parent seed/task/data/optimizer/evaluation protocol is inherited. Failure dumps include both fallback trajectories and pre-update metadata. No /tmp or sitecustomize dependency.

Candidate GPU checks transfer by byte-identical solver/core/trust implementation and import-only step normalization. Formal preflight records the transfer evidence. This round does not launch train/evaluate or create a formal recovery slot. Diagnostic POST_STEP_STATE files support a direct single-step restore; they are not canonical epoch checkpoints. Before a later authorized continuation, use the existing parent-child canonical recovery migration; do not continue a failed process or label a mid-epoch diagnostic state an epoch recovery slot.

## Authorized epoch continuation

The canonical v3 task8 epoch10 boundary (global step41799) is migrated into the
v4 recovery directory. Its model, AdamW, scheduler, RNG, modes, counters and 211
accepted epoch records are unchanged. The immutable parent copy remains separate
from the two new rotating recovery slots. Diagnostic pre/post-failure states are
not used. Parent task6/7 ep99 endpoints are linked read-only for native evaluation.

`EXECUTION_FROZEN.json` is an execution-only addendum to the unchanged scientific
protocol and preflight. It freezes the launch, migration and epoch observer code.
It verifies runtime/config/evaluation sources; native cache/reference hash-on-load
checks remain enabled. It does not claim a new whole-repository integrity audit.
Solver/core/step source bytes are unchanged from v4 preflight.

Launch once after migration:

```
source .migration/activate.sh
nohup python -u -m sample_qp_lifelong_v1_continuation_v4.queue > reports/sample_qp_lifelong_v1_continuation_v4/queue_formal.log 2>&1 < /dev/null &
```

The epoch11 observer verifies B185 uses only the first fallback, B258 uses the
second fallback, both satisfy unchanged KKT/writeback limits, and all309 batches
finish. It reloads the new canonical checkpoint and compares the complete state.
Only then it emits `V4_NUMERICAL_RECOVERY_PASS.json`; training proceeds without
another pause. Any later solver failure dumps and fail-stops. No automatic retry.

Queue phases: task8 remainder, task9 all100 epochs, unchanged ep99 lifecycle
matrix (10 cells,20 trials each), summary. Read `QUEUE_STATUS.json`, `STATUS.json`,
`RUNTIME_RESTORE.json` and the epoch recovery marker for progress. A queue failure
is recorded in `QUEUE_FAILURE.json` and the phase log; never resume failed memory.
