# v4 numerical continuation runtime

Training remains stopped. Standard QP and v3 L-BFGS-B are unchanged. Only a failed first fallback invokes trust-constr from the original 30k solution, using the first KKT-certified iterate.

The frozen parent seed/task/data/optimizer/evaluation protocol is inherited. Failure dumps include both fallback trajectories and pre-update metadata. No /tmp or sitecustomize dependency.

Candidate GPU checks transfer by byte-identical solver/core/trust implementation and import-only step normalization. Formal preflight records the transfer evidence. This round does not launch train/evaluate or create a formal recovery slot. Diagnostic POST_STEP_STATE files support a direct single-step restore; they are not canonical epoch checkpoints. Before a later authorized continuation, use the existing parent-child canonical recovery migration; do not continue a failed process or label a mid-epoch diagnostic state an epoch recovery slot.
