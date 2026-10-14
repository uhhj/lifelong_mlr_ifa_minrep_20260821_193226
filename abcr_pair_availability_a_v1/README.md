# ABCR pair availability: CPU experiment A

`original_probe.py` is the byte-identical user-provided script (SHA256
`e03d3075d5925781fde81ebb1474f8806a18a746379dffe67b327e3b0a4be0c8`).
Its thresholds, IQR scaling, two-view descriptors, last-action labels, gripper
rule, top-k procedure and static-action definition are unchanged.

The wrapper verifies surviving OBJECT feature files, native replay membership,
and stage separation before calling the script. Seed42 uses committed stage
buffers; seeds43/44 use deterministic reconstructions from native sampling
helpers and original configurations, explicitly distinguished in the manifest.
Task6 has no past incremental-task replay. No missing GOAL/LIBERO50 data are
synthesized. No trajectories are guessed from exported record IDs.

Additions are reporting and execution controls only:

- CPU-only, one torch/BLAS thread, no model/environment imports.
- 500 batch simulations as in the supplied study plan, rather than CLI default300.
- A paired comparison of random5+5 and five neighbors per current anchor.
- Explicit retrieval search cost, unique windows and useful old-task coverage.
- Fail on invalid inputs rather than silently dropping records.
- Input/code provenance and atomic completion per stage; resume completed stages
  only after verifying input/code/parameter/runtime/provenance bindings and output
  hashes. No training state is modified.

Run from the repository using its existing environment:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest abcr_pair_availability_a_v1.tests -v
python -m abcr_pair_availability_a_v1.run
```

Outputs: `reports/abcr_pair_availability_a_v1/`. The official training sampler
does not guarantee5+5 batches; this is the requested availability simulation.
Retrieval may use25 distinct old windows versus5 for random, so this is not an
equal-search-budget method comparison. Neither candidate counts nor overlapping
window pairs establish physical-state comparability, independent sample size,
ABCR training benefit, or a decision to train ABCR. Experiment B is not run.

## Resume binding fix (after original result commit8409470)

Schema2 completion records bind the ordered current/replay file membership and
SHA256 bytes, actual package/native-helper code, full shared parameters, runtime
identity and stage provenance, as well as every output. Inputs are checked again
before publishing completion. Input/stage manifests are immutable; a mismatch
raises without replacing historical records. Same-size/same-mtime input changes
are detected. Missing binding in a legacy completion record is an error, not an
implicit upgrade. The whole output directory is protected by a process lock.

Use a **new** output directory when running patched code:

```bash
python -m abcr_pair_availability_a_v1.run \
  --root /data/lifelong_mlr_ifa_minrep_20260912 \
  --output /data/lifelong_mlr_ifa_minrep_20260912/reports/abcr_pair_availability_a_v1_review_v1/reproduction
```

On an unchanged completed run, all bindings and output hashes are checked and
the process returns without rewriting reports. A changed input, code, protocol,
runtime or output fails closed; use another output directory for a new study.
The user-supplied original script remains byte-identical. `settings.py` supplies
the same fixed parameters to it and to `batches.py`, with consistency assertions.
