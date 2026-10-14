# Real-buffer R1–R4 validation

This independent CPU experiment never changes formal MLR, buffers, checkpoints,
QP, or training queues. Read `reports/abcr_real_buffer_validation_v1/PROTOCOL.json`
and `FINAL_REPORT.md` for definitions and scope limits.

Dependencies are the existing project environment. Do not install or upgrade.
The server checkout contains unrelated local changes; execute this package from
its isolated runtime and pass the original checkout through `--repo`.

Execution order:

```bash
python -m abcr_real_buffer_validation_v1.prepare --repo REPO --out OUT
python -m abcr_real_buffer_validation_v1.tests
python -m abcr_real_buffer_validation_v1.analyze --repo REPO --out OUT
python -m abcr_real_buffer_validation_v1.audit_pack --out OUT
# Review the blinded images; save honest AI_SEMANTIC_LABELS.csv provenance.
python -m abcr_real_buffer_validation_v1.gradient_probe --repo REPO --out OUT
python -m abcr_real_buffer_validation_v1.batch_coverage --repo REPO --out OUT
python -m abcr_real_buffer_validation_v1.verify --repo REPO --out OUT
python -m abcr_real_buffer_validation_v1.report --out OUT
```

Preparation validates original input hashes and rejects ambiguous identities.
The frozen bindings reject changed source/input/protocol. R4 resumes only matching
completed pair records; it never replays preceding training batches. Analysis is
deterministic but is not a background queue: do not run it again on a completed
report directory merely to regenerate a report. `report` alone reads measurements.

R1 feature filenames are shuffled export indices, not trajectory times. Exact
eight-action byte sequences establish lineage. R4 loads native class files and
unchanged native method ASTs, avoiding the broken top-level LIBERO package layout
without editing its source. Both complete historical states pass strict model,
optimizer, scheduler, counter, CPU RNG and mode restoration. Historical CUDA RNG
is retained in the source, but this CPU experiment makes no GPU trajectory parity
claim. Independent native `observe` parity checks the actual candidate adapter.

Large arrays/images stay in the server report directory; image paths, hashes and
raw HDF5 identities are recorded. This is not a self-contained downloadable data
release. `WINDOW_IDENTITY_MANIFEST.json.gz` is an exact compressed copy of the
server JSON; decompress when preparing a local evidence-only review. The prior
P1 raw/features and original `.pt`/HDF5 assets are required for full re-execution.

The semantic audit is **AI**, not human annotation. Its diversity-oriented sample
is not an unbiased estimate of population semantic precision. R4 has only five
eligible pairs on two shared anchor states; no strong inference is justified.
