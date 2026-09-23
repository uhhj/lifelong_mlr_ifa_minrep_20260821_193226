# QP-corrected update fidelity and conditional functional worksets

Isolated diagnostic using immutable actual_update_fidelity_v1 native MLR states. No formal training resumes. The existing v4 QP is imported directly; no solver edits.

Activate `.migration/activate.sh`. Frozen manifests and protocol are in `reports/qp_corrected_update_fidelity_v1/`. Run `python -m unittest qp_corrected_update_fidelity_v1.tests -v`, then `python -m qp_corrected_update_fidelity_v1.run --phase all`. The queue wrapper verifies targeted source/manifest hashes, launches the worker, and commits/pushes only this package and reports on exit.

Resume skips only starts with atomically written COMPLETE.json; rolling full diagnostic states live under `/root/qp_corrected_update_fidelity_v1/`. Original full-state witnesses are checked at each start. Do not use those rolling diagnostic files to continue formal training.

Phase1 reuses120 starts, 24old and24new probes, and literal formal references32/task.122 old windows overlapped references and were replaced before measurements, by identity only.2758 original probes were preserved. Most old probes share demonstrations with references; report window and trajectory independence separately.

Six native-candidate parity checks and32/64 real-QP smoke checks precede measurement. GMM loss uses deterministic eval and learned variance. Actual displacements are exact float64 differences between float32 model states; formal QP still receives its original float32 displacement. Source optimizer/scheduler/RNG are restored after every diagnostic.

Phase2 executes only if both protection and corrected Taylor fidelity are supported and all numerical starts succeed. Head-only mean-action GN rank32 and frozen action scaling are used solely for redundancy. All reduced problems use the same v4 solver; full constraints are checked and violated real constraints added. Costs include selection/geometry and repeated solves. No hypothetical Phase2 PASS is reported.

Progress: STATUS.json, QUEUE_STARTED.json, run.log, per-start COMPLETE.json. Scientific conclusions: SUMMARY.json and FINAL_REPORT.md. End-of-run upload: PUSH_STATUS.json. Failed solver starts remain in all accounting.
