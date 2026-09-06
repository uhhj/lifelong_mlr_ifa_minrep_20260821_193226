# Server execution

Baseline: d75fb371cf575beb079ce064d70d2ae323da95dd.
Branch: codex/aggregate-replay-constraint.
Server: dm-MS-7D70; repository /media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep.
Python: /home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python; unchanged baseline environment.

Installation dry-run and installation succeeded. Unit tests: 15 passed, one optional
local-source test skipped. Real-policy native-vs-shadow parity passed exactly for
new-only, mixed, old-only batches, including RNG and optimizer state.
Both single-GPU torchrun smoke arms completed task6/7 and wrote checkpoints.
Task7 had 3 reference batches out of 4 in both arms; no conflict occurred in these
four batches, so smoke does not itself demonstrate a fired projection. Synthetic
projection tests passed. Formal diagnostics record actual projection frequency.

Formal queue uses GPU0 only, seeds42/43/44 serially, 100 epochs and fixed ep99 full
matrix evaluation with 20 trials. Existing matched MLR and corrected IFA results
are reused. Completion/configuration/finite-loss checks do not gate on performance.
Failure stops the queue without overwriting or retrying a nonempty run.

Launch: `nohup bash replay_constraint_experiment/run_queue.sh > runs/replay_constraint_driver/queue.log 2>&1 < /dev/null &`
Status: `runs/replay_constraint_driver/status.txt`.
This record documents readiness and launch protocol, not completed formal results.
