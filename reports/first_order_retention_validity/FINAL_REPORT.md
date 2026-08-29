# First-Order Retention Validity Diagnosis

## Identity

- Verdict: `LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED`
- Layer attributions: `FINITE_TRAJECTORY_FIRST_ORDER_BREAKDOWN`
- Branch: `research/first-order-retention-validity-diagnosis`
- Starting SHA: `f51e10436d03404804068831acb81ceeaa2d249d`

## V2 closure

The V2 negative result was committed and pushed before this diagnosis as `f51e10436d03404804068831acb81ceeaa2d249d`. Its preregistered verdict remains `PER_TASK_TEMPORAL_ROBOT_IGFA_NOT_SUPPORTED`.

## Protocol

This is an offline latent/checkpoint diagnosis. It reused frozen V1/V2 checkpoints, seed-specific feature caches, temporal sidecars, and deterministic distribution-matched batches. No optimizer step, retraining, or rollout evaluation was performed. Checkpoint displacement includes the realized AdamW training trajectory, while the checkpoint-A Taylor prediction remains only a start-point first-order proxy.

## Layer A

Primary rates below use stages 8-9; stage 7 was retained as a single-old sanity check.

- V1 head composition leakage: 4.66%
- V2 head composition leakage: 4.62%
- V1 transformer flip: 0.00%
- V2 transformer flip: 0.00%
- V1 full first-order unsafe: 8.13%
- V2 full first-order unsafe: 5.97%

## Short checkpoint transitions

- Cases: 180
- V1 head false-safe / transformer flip / full false-safe: 56.67% / 0.00% / 56.67%
- V2 head false-safe / transformer flip / full false-safe: 50.00% / 0.00% / 50.00%
- V1 sign agreement; Pearson/Spearman: 43.33%; 0.6402/0.6181
- V2 sign agreement; Pearson/Spearman: 50.00%; 0.7152/0.6864
- Per-seed full false-safe (both methods): seed42=50.00%, seed43=51.67%, seed44=58.33%

## Long horizon

- V1 full false-safe / sign agreement: 94.44% / 5.56%
- V2 full false-safe / sign agreement: 88.89% / 11.11%

## V1 vs V2

V1 and V2 use identical probes and checkpoint intervals. Differences therefore describe their realized local geometry and checkpoint trajectories rather than probe resampling.

## Seed44 task7

The dedicated CSV contains 20 short transition cases; each method/stage subset contains five preregistered intervals.
- V1 stage8 old7: full false-safe 60.00%, transformer flip 0.00%, sign agreement 40.00%
- V1 stage9 old7: full false-safe 60.00%, transformer flip 0.00%, sign agreement 40.00%
- V2 stage8 old7: full false-safe 60.00%, transformer flip 0.00%, sign agreement 40.00%
- V2 stage9 old7: full false-safe 60.00%, transformer flip 0.00%, sign agreement 40.00%

## Observed

Layer-A signs, checkpoint predictions, actual BC-loss changes, and displacement norms are direct outputs of the frozen formal artifacts.

## Inferred

H8 is classified as `LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED` under the preregistered descriptive thresholds. Layer attributions are `FINITE_TRAJECTORY_FIRST_ORDER_BREAKDOWN`.

## Not established

A false-safe result does not uniquely identify Hessian curvature; gradient-field rotation, optimizer trajectory, and multi-batch nonstationarity remain possible contributors. This diagnosis also does not establish how BC-loss drift maps to rollout-success retention.

## Verdict

`LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED`

## Layer attributions

- `FINITE_TRAJECTORY_FIRST_ORDER_BREAKDOWN`

## Next action

Test a finite-step loss-change or functional-loss probe criterion before adding further hard gradient constraints.
