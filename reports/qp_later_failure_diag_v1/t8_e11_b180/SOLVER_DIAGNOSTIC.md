# Offline solver diagnostic: task8 / epoch11 / Python batch185

The directory name remains `t8_e11_b180` for historical continuity; the captured failing update is Python batch **185**, immediately after successful batch184, at pre-update global step **41984**. This report uses only the saved QP arrays. No training replay was run. `formal_training_resumed = false`.

## Asset and residual validation

`before_extended.npz` and `after_extended.npz` have exactly equal `K`, `f`, `A`, `b`, `u_norm`, `valid`, and `scale`. All arrays are finite; the problem is 64-dimensional. The normalization `scale=sqrt(diag(K)); A=K/scale[:,None]/scale[None,:]; b=f/scale/u_norm` is reproduced exactly (maximum errors are 0). A symmetry error is 2.22e-16; normalized diagonal range is [0.9999999999999998, 1].

Using `p=b-A@x`, the independent float64 KKT is `max(max(p,0), max(-x,0), max(abs(x*p)))`. It reproduces before KKT **4.864608038612772e-08** and after KKT **2.301811075078319e-08**. Decimal precision-80 recomputation from exact binary64 inputs gives before **4.864608033267000818180995451280390630457664552824878711664022875454937989E-8** and after **2.301811074094009888398297799523533138442605079667889412412885071640422653E-8**. Neither crosses the fixed `1e-8` decision threshold. Absolute float64-versus-80-digit KKT differences are before `5.345771315820720588794434656E-17` and after `9.843095023426462730806462439E-18`.

## Geometry

Eigenvalues range from 4.03184e-07 to 49.8691; there are 0 negative eigenvalues. Condition number is 1.23688e+08; singular values range from 4.03184e-07 to 49.8691. NumPy default rank is 64; relative thresholds `1e-12`, `1e-10`, `1e-8` give ranks 64, 64, 63. The maximum unique off-diagonal `|Aij|` is 0.999999104; pair counts above 0.99, 0.999, 0.9999 are 1248, 1199, 310. This shows strong near-collinearity and rank sensitivity to tolerance, despite a positive computed spectrum.

Before and after each have 13 positive multipliers and 51 coordinates `x<=1e-12`. Maximum x is 0.930774/0.929345; `||x||` is 1.47678/1.47628. Full counts are in `geometry.json`.

## Coordinate curve

Objective was monotone nonincreasing: **True**. KKT was not monotone: it increased through 5k, then decreased. Relative KKT reductions were 52.68% from 0 to 100k, 17.63% from 20k to 50k, and 41.20% from 50k to 100k. Complementarity increased through 50k before dropping at 100k; primal-positive residual declined overall. The eight exact points are in `convergence.csv`.

## L-BFGS-B

The unchanged objective, analytic gradient, and `x>=0` bounds were used with SciPy 1.10.1. `success` is informational; KKT is the acceptance check.

- **From before:** success=True; nit/nfev/njev=91/110/110; 0.0084s. Objective `-0.09709162431154275`; `||x-x_start||=0.00667984`. Float64 KKT `2.592970393955341e-09`; precision-80 KKT `2.59297052290799026634670635764025130241429037510838239216832447342486748E-9`. Both pass `1e-8`.
- **From after:** success=True; nit/nfev/njev=12/16/16; 0.0012s. Objective `-0.09709162429436269`; `||x-x_start||=4.50253e-05`. Float64 KKT `2.051244923917395e-08`; precision-80 KKT `2.051244914484720528324276674114592067494259084551411273370565696572143107E-8`. This start does not pass despite optimizer success.

The strict result from the before start is saved in `candidate_strict_solution.npz`, bound to the exact `A` and `b` hashes. `trust-constr` was not run because one L-BFGS-B result already passes both independent KKT checks, consistent with the conditional protocol.

## Decision

**`COORDINATE_POLISH_TOO_SLOW`.** The same QP has a strictly acceptable solution under the unchanged `1e-8` criterion, while coordinate polish remains above it after 100k added sweeps. This supports investigating an equivalent solver fallback; it does not validate parameter writeback or authorize training. The next step, if approved, is a separate single-update test from `PRE_FAILURE_STATE.pt` with the frozen batch185 input, checking writeback residual and exactly one optimizer/scheduler step. Formal training remains stopped.
