# V4 formal continuation started

- Implementation commit: `86d7b4103c026472129fa3b74a2e349ea24a4842`.
- Formal parent: v3 task8 epoch10 boundary, global step41799; no diagnostic snapshot used.
- Full restore: model, AdamW, scheduler, RNG, counters and modes exact.
- Task8 epoch11 completed: global step42108,309 successful batches,587.73 seconds.
- B185: L-BFGS-B PASS,KKT4.1823128293927425e-10; trust-constr not called.
- B258: trust-constr first-pass iteration40,KKT3.3966850332638125e-9; physical residual0.
- B258 counters:3657->3658 for optimizer and scheduler; following50 batches completed.
- V4_NUMERICAL_RECOVERY_PASS: true; canonical checkpoint reload exact for all state components.
- Epoch11 checkpoint: `/root/sample_qp_lifelong_v1_continuation_v4/recovery/slot1.pt`.
  This is a rotating canonical recovery slot, not an immutable endpoint. Consult RECOVERY_INDEX for the latest boundary.
- Formal training resumed: true. No solver/KKT/coordinate/optimizer/protocol settings changed.
- Queue PID3128; training PID3130. Continuing task8 remaining epochs -> task9 100epochs -> ep99 lifecycle evaluation.
- Frozen evaluation: seed46,10 cells x20 trials, native task order/action sampling/success semantics.
- Any future solver failure saves QP_RESULT and stops; no tuning and no automatic retry.

The experiment remains in progress. This verifies numerical recovery, not task8/task9 completion or closed-loop performance.
Scientific protocol and preflight reports are retained unchanged; the execution-only addendum records migration, scoped integrity and observation code.
