# Task8 epoch12 Python batch58: offline working-set refinement

## Scope

Original `ec18946` QP arrays and JSON were verified against the Git blobs. All existing inputs, formal runtime sources and the recovery index remained unchanged.
`training_active=false`. No model was loaded, no parameter writeback, no formal v5 and no training restart.
KKT stays at1e-8. QP, normalization, constraints and lambda mapping are untouched. Precision is80 decimal digits from exact binary64 values.

## Failure mechanism

| Stored candidate | Float64 KKT | Dominant coordinate/component |
|---|---:|---|
| Standard30k | 1.8797007229e-7 | coordinate29 complementarity; coordinate15 also violates primal positivity |
| Original L-BFGS-B | 1.8797004880e-7 | same complementarity bottleneck at29 |
| Trust final | 6.7897659823e-8 | primal positivity at40; complementarity also fails |
| Trust minimum-KKT iterate56 | 1.9770926601e-8 | complementarity at29; still FAIL |

Indices are0-based. Full exceeding-coordinate lists, x and primal values are in `B58_RECOMPUTATION.json` and `B58_COORDINATE_RESIDUALS.csv`.

The standard result already has the same33-variable positive support as the certified refined solution. One direct stationary solve on this support passes all64 original coordinates without adding or removing a variable. Thus this instance is consistent with insufficient stationarity accuracy inside an already correct working set, not an unattainable1e-8 gate.
At coordinate29, x≈13.2243 multiplies a stationarity residual≈-1.4214e-8 into complementarity≈1.8797e-7. The trust final point has a different dominant residual; its internal xtol success does not certify the project KKT.

## Geometry of this instance

- A symmetry max error: 2.220446049250313e-16.
- cond(A): 42561752.2858; eigenvalue range: [6.5794383138e-07, 28.0032423593].
- Numerical ranks: default64; relative1e-12/1e-10/1e-8: {'1e-12': 64, '1e-10': 64, '1e-08': 64}.
- Maximum unique off-diagonal absolute correlation: 0.999997091889.
- Unique pair counts: {'0.99': 172, '0.999': 95, '0.9999': 33}.
- Standard/L-BFGS-B33-variable working-set condition number:1.4554027486e7.
- Trust start support contains all64 variables, condition number4.2561752286e7;31 boundary removals recover the same33-variable support.
- All actual working-set matrices and condition numbers are recorded in candidate TRACE.json. No real fixture encountered a singular linear solve.
- Spectrum uses the symmetric part for reporting only; actual linear solves and KKT use original unsymmetrized A.

## Four fixed B58 starts

| Start | Working-set iterations | Linear solves | Seconds | Float64 KKT | Decimal80 KKT |
|---|---:|---:|---:|---:|---:|
| x_standard | 1 | 1 | 0.017099 | 3.10008e-14 | 2.31411e-14 |
| lbfgsb_candidate | 1 | 1 | 0.012794 | 3.10008e-14 | 2.31411e-14 |
| trust_final | 32 | 32 | 0.145301 | 3.10008e-14 | 2.31411e-14 |
| trust_best_kkt_iterate | 32 | 32 | 0.141230 | 3.10008e-14 | 2.31411e-14 |

The minimum-KKT trust iterate was used only as an input; it was never accepted as-is. All four starts reached byte-identical saved refined x.

## Same-rule regression

These are diagnostic refinements from the original standard path, not a newly approved formal fallback order. F1 is already accepted by that path and is returned bit-for-bit unchanged.

| Fixture | Standard KKT | Refinement iterations | Seconds | Float64 KKT | Decimal80 KKT | Result |
|---|---:|---:|---:|---:|---:|---|
| F1 | 6.92311e-09 | 0 | 0.012154 | 6.92311e-09 | 6.92311e-09 | UNCHANGED_STANDARD_PASS |
| F2 | 1.00388e-08 | 1 | 0.009516 | 1.11022e-16 | 5.35423e-17 | STRICT_PASS |
| B185 | 4.86461e-08 | 1 | 0.009504 | 2.22045e-16 | 2.00869e-16 | STRICT_PASS |
| B258 | 1.00936e-08 | 1 | 0.009540 | 2.22045e-16 | 2.36376e-16 | STRICT_PASS |
| E12_B58 | 1.8797e-07 | 1 | 0.012299 | 3.10008e-14 | 2.31411e-14 | STRICT_PASS |

Timing includes project KKT/Decimal checks inside refinement, but excludes input reading, source hashing, geometry reporting and prior standard solving. F1 standard-path regeneration separately cost3.848931seconds; F1 refinement cost is verification-only. These are single CPU timing observations, not end-to-end training speedups.

## Verification and limits

-8 unit tests passed, including exact-active-set enumeration for12 synthetic4D strictly convex QPs, removal/re-entry, singular handling, deterministic replay, budget and cycle detection.
- Formal float64 checker matches independently implemented float64 components for every saved refined candidate.
-9 saved candidates additionally passed an independent exact-rational KKT audit. All original64 coordinates were checked; Decimal80 results agree to less than1e-65.
- No ridge, pseudoinverse, rank truncation, eigenvalue clipping or permanently removed constraints.
- No unresolved real fixture in this set. Singular/degenerate working systems remain an explicit fail-stop case; only synthetic singular handling was exercised.
- Working-set re-entry is implemented and unit-tested; these real starts needed zero re-entries. Five selected historical fixtures do not prove robustness on all future QPs.
- No actual float32 writeback, optimizer/scheduler/RNG parity or continuation validation was performed in this round.

`training_active=false`.
