# New v3 QP failure capture

Formal training remains stopped (`formal_training_resumed=false`). No solver settings, QP, constraint, threshold, or formal recovery slot were modified.

- Capture smoke: PASS; same function objects and explicit diag_dir throughout step -> core -> solver.
- Actual failure: task8 / epoch11 / Python batch258 (ordinal259), pre-update global step42057. Location comes from successful event rows, not periodic STATUS.
- Parent: verified post-batch185 state, global41985. Restored model, AdamW, scheduler, modes and all RNG state.
- Replayed batch186..257: 72 successful updates; every recorded non-timing field matches the historical event exactly.
- Saved pre-batch258 state with immutable batch/reference identities and per-reference tensor fingerprints. Save/reload leaves state unchanged.
- Failed batch calls native AdamW/scheduler once: counters3657 -> 3658. Its partially updated in-memory state is not resumable; use saved PRE_FAILURE_STATE.
- before_fallback.npz and after_fallback.npz are atomic, loadable and contain identical K/f/A/b/scaling. Independent binary64 and 80-digit residuals are in FINAL_REPORT.json.

Before fallback KKT: 1.0093558544532755e-08

After fallback KKT: 1.547045713756745e-08

The current L-BFGS-B fallback terminates but does not satisfy the unchanged 1e-8 KKT gate on this instance. This report captures the failure; it does not select a new solver or infer QP infeasibility. All subsequent QP work can use these arrays without prefix replay.

State: `/root/qp_v3_failure_capture_v1/PRE_FAILURE_STATE.pt`

L-BFGS-B stopped after one iteration on relative objective decrease. Its success flag does not imply KKT acceptance. The violating component is primal feasibility, confirmed at 80 digits; no threshold decision flip.
