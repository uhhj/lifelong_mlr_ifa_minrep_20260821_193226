# Offline working-set refinement diagnosis

This package is diagnostic only. It is not imported by any formal training runtime.
Original matrices, solver sources, recovery slots and experimental outputs are read-only.

Inspect `reports/qp_workingset_refinement_diag_v1/PROTOCOL.json` for fixed rules.
The entry point refuses to overwrite an existing protocol. Reproduce into a fresh
output directory without changing the numerical implementation:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m unittest qp_workingset_refinement_diag_v1.tests -v
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -c 'from pathlib import Path; from qp_workingset_refinement_diag_v1 import run; run.OUT=Path("reports/qp_workingset_refinement_diag_v1_reproduction"); run.main()'
```

No GPU/model build or environment preflight is needed. Source paths are fixed to
the five real saved QPs. B58 source hashes are checked against commit ec18946.
F1 alone regenerates the already accepted standard CPU QP path; its result is
returned unchanged. Other cases reuse their saved standard iterates.

The working-set solver has a 512-iteration limit, deterministic entry/exit order,
exact-state cycle detection and explicit singular/unhandled numerical outcomes.
It solves unmodified working-set equations with `numpy.linalg.solve`. Up to three
residual corrections are allowed by the frozen rule; none were needed on the
measured B58 starts. Every candidate is validated against all original coordinates.

`audit.py` independently checks saved candidates using exact rational arithmetic.
`report.py` renders existing evidence only. Do not interpret the offline regression
as validation of float32 model writeback or a new formal continuation runtime.
