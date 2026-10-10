# OBJECT E1-O: full-coverage protective reordering

Frozen execution archive for the completed LIBERO-OBJECT seed42 experiment,
derived from commit `b3590dfe836c035e799ed246f7c8da3dcf85bf64`'s isolated E1-U subtree.
The root GOAL implementation and previous experiments are unchanged.

Every epoch preserves the original full dataset traversal and exact ID counts.
At stage start and every500 updates, at most64 pending same-task old positions
per task are scored and permuted without replacement inside the next500-step
or epoch-end boundary. Current IDs, task slots, first batch slots and incomplete
tail slots are pinned. H16/N16, alpha1/tau1, original BC+IFA, gamma0.3,
OBJECT use_state=False, optimizer/scheduler and random buffer storage are retained.
The E1 replacement hook is disabled.

- `source/`: exact source snapshot; 171 active files match the frozen contract.
  Historical `.hydra/` provenance is retained outside the active hash scope.
- `source.patch`: delta against the archived E1-U subtree, not the root GOAL code.
- `tests/`: target56 software tests; real preflight used exactly5 GPU updates,
  no extra rollout. Publishing does not rerun training or environment tests.
- `USER_PROTOCOL.txt`, `PROTOCOL_AUDIT.txt`: fixed protocol and disclosed gaps.
- `COMMANDS.txt`: historical launch evidence; final exits are in the result report.
- `independent_acceptance.py`: external read-only completion verifier.
- [Completed results and evidence](../../reports/object_seed42_coverage_order_E1O/README.md).

Actual execution used PyTorch1.13.0+cu116 on RTX4080 SUPER. Server paths and external
dependencies remain in the frozen files. This is an execution archive, not a
standalone installer; do not run against a completed output directory.
Weights, latent features, demonstrations and buffer tensors are excluded.
