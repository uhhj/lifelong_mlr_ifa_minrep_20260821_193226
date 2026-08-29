# Finite-Step Functional Loss Probe Validity Diagnosis

## Identity

- Verdict: `FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_NOT_SUPPORTED`
- Signal horizon: `WEAK`
- Branch: `research/finite-step-functional-probe-validity`
- Starting SHA: `e49c1e1e9b4188528f457de29432c54df75f99af`

## Prior evidence

The prior diagnosis established `LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED` with `FINITE_TRAJECTORY_FIRST_ORDER_BREAKDOWN` attribution.

## Protocol

The analysis reused the committed 180 short cases, 36 long-horizon cases, 2,304-record probe manifest, endpoint losses, and first-order predictions. It evaluated only the frozen policy head and temporal transformer along the realized checkpoint chord at fixed formal horizons 0.05/0.10/0.25/0.50. No training, optimizer step, rollout, endpoint recomputation, or probe resampling was performed. Rho=1 was used only for smoke endpoint parity.

## Primary rho=0.10 table

| Scope | FO false-safe | Finite false-safe | Finite balanced accuracy | Finite sign agreement | Rescue rate |
|---|---:|---:|---:|---:|---:|
| V1 | 56.67% | 56.67% | 50.00% | 43.33% | 0.00% |
| V2 | 50.00% | 50.00% | 50.00% | 50.00% | 0.00% |
| Pooled | 53.33% | 53.33% | 50.00% | 46.67% | 0.00% |

## Material-change subset

The fixed `abs(relative_actual_delta) >= 0.005` subset contains 163 cases at rho=0.10. First-order versus finite false-safe is 52.15% versus 52.15%; finite false-alarm is 0.00%, balanced accuracy is 50.00%, and sign agreement is 47.85%.

## Horizon table

| rho | Finite false-safe | Reduction | Balanced accuracy | BA gain | Sign agreement |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 53.33% | +0.00 pp | 50.00% | +0.00 pp | 46.67% |
| 0.10 | 53.33% | +0.00 pp | 50.00% | +0.00 pp | 46.67% |
| 0.25 | 53.33% | +0.00 pp | 50.00% | +0.00 pp | 46.67% |
| 0.50 | 53.33% | +0.00 pp | 50.00% | +0.00 pp | 46.67% |

## Seed replication

| Seed | FO false-safe | Finite false-safe | Reduction | Finite balanced accuracy |
|---:|---:|---:|---:|---:|
| 42 | 50.00% | 50.00% | +0.00 pp | 50.00% |
| 43 | 51.67% | 51.67% | +0.00 pp | 50.00% |
| 44 | 58.33% | 58.33% | +0.00 pp | 50.00% |

## Rescue analysis

At rho=0.10, the pooled rescue rate among committed first-order false-safe cases is 0.00%. The all-case false-safe reduction is +0.00 pp. Among 96 endpoint-harmful cases, 0.00% were detected by rho=0.10 and 0.00% by rho=0.50; 100.00% remained late harmful reversals through rho=0.50.

## Nonlinearity residual

At rho=0.10, mean signed residual `probe_delta - rho*pred_all` is 0.065009; mean absolute residual is 0.065009. This is descriptive nonlinearity evidence, not an identified Hessian term.

## Long horizon

| rho | Finite false-safe | Reduction | Balanced accuracy | Sign agreement |
|---:|---:|---:|---:|---:|
| 0.10 | 91.67% | +0.00 pp | 50.00% | 8.33% |
| 0.25 | 91.67% | +0.00 pp | 50.00% | 8.33% |
| 0.50 | 91.67% | +0.00 pp | 50.00% | 8.33% |

## Seed44 task7

- V1: first-order 60.00%; rho0.10 60.00%; rho0.25 60.00%.
- V2: first-order 60.00%; rho0.10 60.00%; rho0.25 60.00%.

## Observed

Intermediate functional losses and their classifications are direct evaluations on the frozen, committed probe records along the realized checkpoint chord.

## Inferred

H9 is `FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_NOT_SUPPORTED` under the preregistered rho=0.10 thresholds. The descriptive signal horizon is `WEAK`.

## Not established

The realized checkpoint chord uses the future endpoint direction, so it is not an online-deployable criterion and is not the actual optimizer path. The diagnosis does not evaluate rollout success and does not prove Hessian causality; gradient-field rotation, optimizer dynamics, and multi-batch nonstationarity remain possible contributors.

## Verdict

`FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_NOT_SUPPORTED`

## Signal horizon

`WEAK`

## Next action

Model the realized training path or test old BC-loss drift against rollout-success forgetting.
