# QP corrected-update local mechanism

Formal training remains stopped. This is a paired single-step diagnostic.

```json
{
  "status": "PARTIAL",
  "starts": 2,
  "solver_failed": 0,
  "QP_PROTECTION": "INCONCLUSIVE",
  "POST_QP_FIRST_ORDER_FIDELITY": "INCONCLUSIVE",
  "PLASTICITY_COST": "MATERIAL_COST",
  "SAMPLE_QP_MECHANISM": "NOT_ESTABLISHED",
  "phase2_gate": false,
  "phase2_status": "NOT_RUN_GATE_NOT_PASSED",
  "formal_training_resumed": false
}
```

Primary paired median positive-damage difference: -0.05852286020914714; start-cluster 95% CI: [None, None].
Prevented windows: 2; newly harmed: 2. Rates retain their distinct denominators.
Native-improving starts: 2; median retained new progress: 0.770054073352997; reversal: 0.0.

|seed|transition|starts|solver failures|median old damage B-A|mean new cost|
|---|---|---:|---:|---:|---:|
|45|6->7|1|0|-0.02511302630106609|0.12471588452657063|
|45|7->8|1|0|-0.0919326941172282|-0.045900742212931334|

## Interpretation and required answers
1. Independent-window protection: INCONCLUSIVE. The primary outcome is positive damage, not reference feasibility.
2. 2/6 cells have negative median paired damage; all six cells are shown, including failures.
3. Reference results and held-out windows/trajectories are separated in REFERENCE_VS_PROBE_REPORT. Exact overlap was removed before outcomes; 122/2880 prior probes replaced by a deterministic identity-only rule. Most remaining probes share reference demonstrations, limiting trajectory generalization.
4. Corrected-update Taylor fidelity: INCONCLUSIVE; medians and paired decline from native d0 are in POST_QP_FIDELITY_REPORT.
5. Actual float32-written displacements, norm ratios, cosine and correction ratios are recorded per start, distinct from ideal solver correction.
6–7. Prevented and introduced harm counts/rates above must be considered together.
8. New-task cost: MATERIAL_COST, with per-cell costs and discernibly improving-start retention.
9. These measurements can reveal a local stability/plasticity tradeoff; they cannot explain cumulative training or rollout behavior.
10. Mechanism status: NOT_ESTABLISHED. If supported with material cost, next study is paired 5/20 steps, not full lifelong training.
11–16. Phase 2: NOT_RUN_GATE_NOT_PASSED; conditional gate is False. No GN sharing/conflict gate is used.

## Limits
- Development seeds45/46/47; source stages cover epochs30–99, not early epochs.
- 120 starts share carriers and probes; confidence intervals cluster by start within cells, not by window. Carrier/trajectory dependence still limits broader inference.
- Formal v4 reference sampling is reused literally, including its historical fixed46 RNG salt. Constraints count32/64, not a fabricated64 for every start.
- No formal training, rollout, optimizer change, KKT relaxation, fallback change, or formal asset mutation.
- Numerical failures remain in the denominator and block Phase2; failed corrected outcomes are missing, never zero.
