# Completed LIBERO-OBJECT E1-O, seed42

**COMPLETED; independent acceptance PASS.** 400epochs,105500 optimizer steps,
28 prescribed checkpoints and880 formal episodes (200 primary,680 secondary).
Pipeline:2026-10-09 21:43:38 to2026-10-10 02:29:01 Asia/Shanghai,4h45m23s.
Formal pipeline and independent acceptance both exited0. No technical rollout
retry, baseline retraining or additional seed was performed.

| Primary epoch99 metric | E0 | E1 | E1-U | E1-O |
|---|---:|---:|---:|---:|
| AUC ↑ |77.708333|77.291667|73.645833|62.395833|
| FWT ↑ |77.500000|76.250000|76.250000|63.750000|
| NBT ↓ |−0.277778|−3.333333|7.222222|1.666667|

E1-O−E0 primary AUC is **−15.3125 points**. The prespecified improvement condition
was not met; seeds43/44 were not run and monitoring was paused. Protocol completion
does not establish algorithm effectiveness.

| Completed stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
|6|35|—|—|—|
|7|35|95|—|—|
|8|20|100|95|—|
|9|5|100|100|30|

Secondary AUC72.5, FWT78.75, NBT11.111111111; selected epochs80/99/99/70.
The secondary endpoint does not replace the primary endpoint.

## Evidence

- [Full Chinese report with all four versions and tasks](FINAL_REPORT.txt),
  [final receipt](FINAL_RECEIPT.json), [frozen contract](evidence/CONTRACT.json).
- [Independent acceptance](evidence/INDEPENDENT_ACCEPTANCE.json),
  [pipeline commands/exits](evidence/commands.log),
  [final process snapshot](evidence/FINAL_PROCESS_SNAPSHOT.json).
- [200 primary episodes](evidence/trials_primary.jsonl),
  [680 secondary episodes](evidence/trials_secondary.jsonl),
  [primary attempts](evidence/attempts_primary.jsonl),
  [secondary attempts](evidence/attempts_secondary.jsonl).
- [Coverage/order audit](evidence/COVERAGE_ORDER_AUDIT.json),
  [movement diagnostics](evidence/order_movement.json).
- [Full lightweight evidence archive](final_light_evidence.tar.gz):474 evidence
  files plus manifest, including400 epoch order logs and complete training stdout.
  Original bytes are verified by [download manifest](evidence/FINAL_DOWNLOAD_MANIFEST.json)
  and [archive receipt](archive_receipt.json). Directly browsable evidence excludes
  the400 order logs and large training stdout, which remain in the archive.
- [Frozen source, tests and protocol](../../experiments/object_seed42_coverage_order_E1O/README.md).

All400 epochs passed exact ID Counter and original sampler reconstruction checks;
1052400 actual slots preserved task/current/locked-position constraints.
181 schedules used24032 candidate gradients under the24896 cap;23458 positions
moved across update steps, mean absolute movement58.016728 steps. Two task sets
were skipped for too few positions. Reordering occurred but overall performance
did not improve. Inputs and original33080 experiment files retained their hashes.

Training46.26min, primary evaluation53m13s, secondary3h05m24s. E0 usedRTX4090,
E1/E1-U RTX3090, E1-O RTX4080 SUPER. Single-seed and hardware differences limit
causal claims; E0 historical wall times and stage-buffer snapshots remain missing.
Comparisons to E1/E1-U change multiple mechanisms. At the later final audit,
nvidia-smi exited9; completed GPU training/evaluation had exited0. Read-only
acceptance used file hashes and CPU index arithmetic, not CPU rollout fallback.

Raw launch and intermediate files retain their original statuses. The final
authority is INDEPENDENT_ACCEPTANCE.json and FINAL_RECEIPT.json. Publication was
explicitly authorized after completion; historical no-push instructions are retained
as execution-time restrictions. No model weights or training data are published.
