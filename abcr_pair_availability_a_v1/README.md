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
  only after verifying their output hashes. No training state is modified.

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
