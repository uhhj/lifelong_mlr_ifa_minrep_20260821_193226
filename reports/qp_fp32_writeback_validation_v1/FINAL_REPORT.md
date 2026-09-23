# FP32 QP physical-writeback validation v1

Stage A uses only the two frozen complete source states; no preceding batches replayed. Solver/QP/formal runtime and prior results remain unchanged.

|state|method|max positive normalized residual|violations|physical pass|write error L2|
|---|---|---:|---:|---|---:|
|s45_t7_e98_b195|W0|5.57503590622e-06|1|False|8.05709785049e-07|
|s45_t7_e98_b195|W1|5.57503590577e-06|1|False|8.05709785049e-07|
|s45_t7_e98_b195|W2|5.57491004008e-06|1|False|8.0570978587e-07|
|s45_t8_e99_b94|W0|8.7703276611e-06|9|False|7.93161986167e-07|
|s45_t8_e99_b94|W1|8.77032767118e-06|9|False|7.93161986167e-07|
|s45_t8_e99_b94|W2|8.77026335656e-06|9|False|7.93161986496e-07|

```json
{
  "status": "STAGE_A_COMPLETE",
  "FP32_WRITEBACK_ROOT_CAUSE": "FP32_GRID_LIMIT",
  "SINGLE_TARGET_MATERIALIZATION": "NOT_SUPPORTED",
  "stage_A_decision": "FP32_GRID_FEASIBILITY_NOT_RESOLVED",
  "WRITEBACK_120": "NOT_RUN",
  "QP_PROTECTION": "NOT_REEVALUATED",
  "POST_QP_FIRST_ORDER_FIDELITY": "NOT_REEVALUATED",
  "PLASTICITY_COST": "NOT_REEVALUATED",
  "TRAJECTORY_DISJOINT_PROTECTION": "NOT_RUN",
  "NEXT_STAGE": "PHYSICAL_REPAIR_REQUIRED",
  "stage_B_allowed": false,
  "formal_training_resumed": false,
  "GN_workset_executed": false
}
```

## Answers and limits
1. Source state, native candidate, same K/f/lambda, dstar rates and archived coordinates are verified against forensic artifacts. W0 exact readback residuals reproduce bitwise.
2–4. W1 two-state gate=False; W2 two-state gate=False. Root classification=FP32_GRID_LIMIT. A grid-limit label means nearest-FP32 materialization of these fixed targets failed; it does not claim no FP32 feasible update exists.
5–7. ULP bins, signed/absolute top-coordinate shares, ideal margins and streaming rounding bounds are retained in ULP_ANALYSIS and the three coordinate/constraint CSVs. The requested upward-ULP bound and a maximum-neighbor-gap bound are both reported, because binade boundaries can have asymmetric spacing.
8–13. StageB/C are not run unless preceding gates pass. No120-state or new probe conclusion is fabricated, and prior118-state results are unchanged.
14. Actual materialization wall time, GPU peak increment, current RSS delta, process peak RSS, and full double target bytes are recorded per branch. StageA resource data do not demonstrate end-to-end training cost.
15. The continuous solver has already certified the original QP. This experiment provides no evidence requiring a QP objective/constraint change: solver/QP semantics do not need modification based on this diagnosis. Physical feasibility handling remains unresolved if W1 fails.
16. GN workset is prohibited in this experiment; a failed StageA gate does not authorize it.

## Proposals only; not executed
- P1 physical line search: test the preregisterable fixed alpha sequence 1,63/64,31/32,15/16,7/8,3/4,1/2,0 and choose the largest actually feasible materialized update. Pros: simple, full scan. Risks: quantized feasibility is not monotonic in alpha, plasticity may shrink; zero is a fallback that gives no learning.
- P2 materialize/check/corrective repair: use violated/near-active coordinates to propose a small correction, then recheck every original constraint after materialization. Pros: directly targets observed violation. Risks: further rounding, cycling and budget exhaustion; final full scan is essential.
- P3 quantization-aware feasibility: include Q_theta(d)=FP32(theta+d)-theta in the feasible-update definition. Pros: matches the executed object. Risks: discrete/non-smooth problem and higher implementation/validation cost.
None of these proposals was implemented or tested. No negative margin, threshold change or extra solver was used.

## Measured conclusions

- s45_t7_e98_b195: continuous KKT=7.14758287279e-12; 80.1415% of nonzero correction coordinates are below0.5ULP of the source parameter. 93.7509% of nonzero final displacement coordinates are >=4 source-parameter ULP.
  Failed constraints' top20 absolute-contribution shares are 1.0707%–1.0707%. Their signed top20/net-error ratios range from 0.023826 to 0.023826. This does not support a few-coordinate-dominance explanation.
  11/11 near-active constraints have ideal margin below the requested rounding bound; all actual failures are in that set. Largest near-active margin/bound=1.8334219e-08.
- s45_t8_e99_b94: continuous KKT=8.41504470273e-12; 83.4348% of nonzero correction coordinates are below0.5ULP of the source parameter. 93.1926% of nonzero final displacement coordinates are >=4 source-parameter ULP.
  Failed constraints' top20 absolute-contribution shares are 1.0796%–2.7453%. Their signed top20/net-error ratios range from -0.12201 to 0.080239. This does not support a few-coordinate-dominance explanation.
  23/23 near-active constraints have ideal margin below the requested rounding bound; all actual failures are in that set. Largest near-active margin/bound=5.1521682e-08.

Removing native-then-correction materialization does not repair either state: W1 and W2 both retain the original1/9 failed constraints. The fixed target has insufficient slack for its nearest-FP32 representation. DOUBLE_MATERIALIZATION is not supported as the principal explanation on these states.

Target rounding is determined by target ULP, which can differ from source ULP after a binade change. Error measured in source ULP can therefore exceed0.5 without indicating a faulty FP32 rounding operation. Bounds use target spacing, and the maximum of upward/downward spacing is also retained.

|state|W1 seconds including readback|W1 full double target MiB|W1 peak GPU increment MiB|W0 peak GPU increment MiB|
|---|---:|---:|---:|---:|
|s45_t7_e98_b195|0.00430048|165.051|663.052|331.052|
|s45_t8_e99_b94|0.00431026|165.051|663.052|331.052|

These two timings include diagnostic rereading of actual displacement and are not a throughput benchmark. Memory is affordable on the tested3090 for StageA, but W1 fails physical feasibility; no120-state aggregate cost was measured.

Final answers8–13: NOT_RUN/NOT_REEVALUATED by the frozen StageA stopping rule, including all120-state gates, old/new NLL reevaluation, trajectory-disjoint evaluation and changes to the118 earlier successful points. Historical results remain immutable.

Next step is a separately preregistered physical-feasibility study, not GN workset. No candidate P1/P2/P3 was run. Continuous solver/QP semantics do not need modification on the evidence here; a validated physical realization is still required. This is not a claim that the FP32 feasible set is empty (zero displacement remains feasible).
