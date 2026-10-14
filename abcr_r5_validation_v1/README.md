# R5 validation (OBJECT, CPU)

This is an isolated diagnostic, not an ABCR training implementation installed in the formal runtime.
The protocol is frozen in `common.py` and the report `PROTOCOL.json` before measurement.

The user authorized AI review in place of human review and an investigator-defined exploratory loss.
AI labels must never be called human labels. `AI_REVIEW_DECISIONS.json` contains the actual blinded
multi-frame decisions, not automatic labels inferred from action distance or the physical gate.

## Inputs

Use the exact paths and SHA-256 values in the saved `START_MANIFEST.json` and
`INPUT_AND_RESULT_INTEGRITY.json`. Required assets include the original HDF5 demonstrations,
P1–P3 raw/features/normalization/frozen thresholds, R1–R4 selected pairs, original seed45/46/47
feature plans, and 12 complete AdamW anchors from `actual_update_fidelity_v1`.
The repository does **not** contain these large private/local assets; cloning alone is insufficient.
Do not reconstruct AdamW from model-only weights, add Task9 states, or replay preceding batches.

The runtime reuses `abcr_real_buffer_validation_v1.gradient_probe`: original policy classes and
unchanged source methods extracted by AST. Source hashes and native `observe` parity are recorded.
CPU parity is not historical cross-device trajectory parity.

## Reproduction in the existing environment

```bash
cd /data/lifelong_mlr_ifa_minrep_20260912
source .migration/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Put the isolated package and its previously audited adapter on PYTHONPATH.
export PYTHONPATH=/tmp/abcr_r5_runtime:/tmp/abcr_real_buffer_validation_runtime:$PYTHONPATH
OUT=reports/abcr_r5_validation_v1
python -m abcr_r5_validation_v1.prepare --repo "$PWD" --out "$OUT"
python -m abcr_r5_validation_v1.tests
python -m abcr_r5_validation_v1.audit --repo "$PWD" --out "$OUT"
# Review all 72 images before exposing action classes or numerical outcomes.
# The checked-in decisions reproduce THIS review, not a new independent review.
python -m abcr_r5_validation_v1.finalize_review --out "$OUT"
python -m abcr_r5_validation_v1.engine --repo "$PWD" --out "$OUT"
python -m abcr_r5_validation_v1.analysis --out "$OUT"
python -m abcr_r5_validation_v1.verify --repo "$PWD" --out "$OUT"
python -m abcr_r5_validation_v1.report --out "$OUT"
```

In the completed run, visual review and numerical measurement ran independently in parallel;
the visual reviewer did not see numerical outcomes before freezing all decisions.
Resumption of B/C checks the protocol, manifest, numerical source binding and original anchor hashes.
Completed measurements are not regenerated. A mismatch must stop reuse, not overwrite it.
The image-pack resume checks saved image hashes; reuse also requires the final input integrity check
against original HDF5 frames/states. It is not a general cache for modified HDF5 inputs.

## Fixed C definition

`z` is the normalized decoder terminal 512-vector. Old-side embeddings are detached.
Share: `mean(1 - z_new.dot(z_old.detach()))`.
Bif: `mean(relu(z_new.dot(z_old.detach()) - 0.5))`.
Both weights are 0.05. Empty eligible sets add zero. All branches reuse the exact native batch10,
native dropout/RNG and optimizer state. There is no retrieval, pair supplementation, tuning or
formal parameter commit. Native branch parity and transactional restoration are required.

The `verify` module does targeted input checks, direct source-state checks, an independent
augmented least-squares recomputation of every grouped prediction, and an actual unit test run.
It is a second implementation check, not an independent investigator replication.

The final recommendation follows the frozen gates even when negative or inconclusive.
`formal_training_resumed=false`.
