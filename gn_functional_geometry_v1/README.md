# Head-only GN mechanism study

This is an isolated diagnostic, not a training algorithm. It never calls an optimizer or changes formal sources/checkpoints. Dependencies are the existing historical MLR environment (PyTorch 1.13, CUDA 11.6, SciPy), not a new installation.

From the repository root, inherit `.migration/activate.sh`. New, empty report directory execution order:

```bash
python -m gn_functional_geometry_v1.prepare
python -m unittest gn_functional_geometry_v1.tests -v
python -m gn_functional_geometry_v1.freeze
python -m gn_functional_geometry_v1.run
python -m gn_functional_geometry_v1.audit
python -m gn_functional_geometry_v1.render
```

`prepare` and `freeze` refuse to overwrite existing selections/protocol. `run` preserves a completed experiment. On existing measurements, use report/audit/render only; do not resample or tune gates. The original measurement log retains a post-measurement missing-report-module exception; report-only completion used the saved observations and did not rerun GPU measurements.

`engine.py` calls the official BriefPolicy decoder and GMMHead. The historical forward method has a root `self.training` dispatch guard; the diagnostic explicitly calls its temporal encoder with all modules in eval mode and verifies that result against the official dispatch. Dropout and random action sampling are disabled. Functional parameter calls never mutate the baseline head. Diagnostic head is float64, decoder latents float32; native interface differences are recorded.

Full trace uses streamed output-gradient squared norms, never a full Jacobian or parameter-square matrix. Top32 eigenspace uses matrix-free JVP/VJP and two seeded eigensolver runs. The stored basis is float32 with float64 validation and downstream arithmetic.

Important interpretation limits:

- Expected-action GN is not expert-NLL curvature or full GMM Fisher; scale-only parameters are invisible.
- Equal-norm direction probes concentrate the whole historical median head step into one direction. They are not actual AdamW marginal projections.
- Old relation gradients and held-out evaluation are trajectory disjoint. Matched-probe Taylor calibration is secondary only.
- One raw-gradient cosine at one checkpoint cannot support a statistical overall-update prediction comparison.
- H2 pair counts do not represent independent samples; only four build demos.
- Engineering gates and action-scale floor are prospective choices, not universal constants.

Large `OLD_TASK_GN_BASIS.npz` remains on the server, with path/size/SHA in ARTIFACT_MANIFEST.json; do not add it to Git. Small raw CSV/JSON/NPZ, source and logs are versioned. No formal continuation is launched.
