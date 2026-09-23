# H3 scale x geometry diagnostic

Scope: one original-MLR seed45 task8-ep99 checkpoint, task8/task9, head only. No training, writeback or formal code changes. This package depends explicitly on the already committed `gn_functional_geometry_v1` diagnostic loader and its immutable data/basis outputs.

Two geometries: reused top16 mean-action GN, and top16 **model Fisher Monte Carlo estimate**. The latter is the score second moment of the exact original mixture log density with 8 fixed model draws per old build window. Samples are detached and drawn with a private generator. It is neither empirical expert-score Fisher nor claimed to be the expert-NLL Hessian/exact GGN.

Four scales are fixed before outcomes: 1, 1/4, 1/16, 1/64 times previous epsilon0. Direction sign follows projected new-task gradient. Norm is matched per direction, preserving the prior probe definition. Primary predictor uses the evaluation set's own gradient (local derivative calibration); cross-trajectory prediction is secondary, clearly separate.

Run in the existing `.migration/activate.sh` environment:

```bash
python -m gn_h3_scale_geometry_diag_v1.freeze
python -m unittest gn_h3_scale_geometry_diag_v1.tests -v
python -m gn_h3_scale_geometry_diag_v1.run
python -m gn_h3_scale_geometry_diag_v1.report
python -m gn_h3_scale_geometry_diag_v1.audit
```

Freeze refuses overwrite; run preserves completed results. Recompute summaries from existing CSV without rerunning the model. No new dependencies are installed. Large Fisher basis remains on server and is hash-recorded, not Git-tracked.

Interpretation: smaller finite differences need not yet lie in the first-order regime. Failed recovery at 1/64 is not a counterexample to directional differentiation. Better behavior under another eigenbasis would not uniquely identify the functional metric as the cause. One fixed MC draw set and one checkpoint cannot establish general solver/control/closed-loop value.
