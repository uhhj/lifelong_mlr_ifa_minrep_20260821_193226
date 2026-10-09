# Completed OBJECT E1, seed42

**COMPLETED and independently audited.** Four tasks × 100 epochs, 105,500 optimizer
updates, 28 prescribed checkpoints, 200 primary and 680 secondary closed-loop
records. Evaluation and aggregation finished at 2026-10-09 03:53:40 Asia/Shanghai.
E0 was not retrained. The primary AUC did not improve, so the prespecified
condition for seeds43/44 was not met; no tuning or subsequent experiment started.

| Primary epoch99 metric | E0 | E1 | E1 − E0 (percentage points) |
|---|---:|---:|---:|
| AUC ↑ | 77.708 | 77.292 | −0.417 |
| FWT ↑ | 77.500 | 76.250 | −1.250 |
| NBT ↓ | −0.278 | −3.333 | −3.056 |

Differences above are computed before rounding. The original pipeline's rounded
subtractions remain preserved in its receipt. AUC averages each task over its
post-introduction stages, then averages over tasks.

E1 primary success rates (%), 20 episodes in each valid cell:

| Completed stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 35 | — | — | — |
| 7 | 45 | 100 | — | — |
| 8 | 35 | 95 | 85 | — |
| 9 | 25 | 95 | 100 | 85 |

Per-task AUC differences are −2.500, −1.667, −7.500 and +10.000 points for tasks
6/7/8/9 respectively. Secondary checkpoint-selection AUC was 81.771 versus E0
80.000; that predeclared secondary view does not replace the primary endpoint.

## Evidence

- [Selection preference and short-window repetition audit](selection_diagnostics/README.md):
  read-only analysis of 389 candidate pools and 230,710 priority assignments.
- [Complete report in Chinese](FINAL_REPORT.txt): both matrices, all tasks,
  actual commands/exit codes, costs, diagnostics, attempt accounting and limits.
- [Independent server audit](SERVER_INDEPENDENT_AUDIT.json) and
  [its read-only implementation](independent_audit.py).
- [200 primary records](evidence/trials_primary.jsonl),
  [680 secondary records](evidence/trials_secondary.jsonl),
  [all 44 evaluation cells](all_evaluation_cells.csv).
- [Training receipt](evidence/TRAINING_COMPLETE.json),
  [400-epoch log](evidence/run_001/paper_repro_train_curve.csv),
  [learning-rate log](evidence/run_001/lr_protocol.csv),
  [resolved configuration](evidence/resolved_training.json),
  [frozen contract](evidence/CONTRACT.json).
- [Complete lightweight archive](lightweight_evidence.tar.gz): includes the
  per-step selection events, full text logs and all archived lightweight evidence.
  It contains **no** checkpoint, latent, demonstration or buffer tensor payload.
- [Publication hash manifest](PUBLICATION_MANIFEST.json). `FINAL_MANIFEST.json`
  refers to paths in the original local final-evidence directory; those paths are
  preserved for provenance and are not all expanded into this publication.
- [Code snapshot](../../experiments/object_seed42_protective_selection_E1/README.md).

Training took 56.52 minutes; evaluation took 5h45m04s after import recovery.
Scoring plus replacement handling took 293.01 seconds within training. Selection
I/O is included in that timing and must not be added twice. Old-slot assignment
fraction was 50.122%; assigning a sample can choose the same ID as the original.

## Disclosures

The archived E0 ran on RTX4090; E1 ran on RTX3090. Original E0 buffer snapshots and
training wall-time logs were unavailable. All 5,921 recovered feature files match
the historical SHA256 manifest, and stage-buffer membership was independently
reconstructed using the unchanged deterministic storage rule. This is not direct
comparison against missing historical buffer snapshots.

The first evaluator launch failed before environment stepping because spawned
workers lacked the sibling-package binding. An external import bootstrap repaired
it; original source, checkpoints and contract were retained. Three automatically
regenerated `.hydra` output files are explicitly separated from frozen source
input checks in the disclosed read-only finalizer.

The inherited evaluator records checkpoint paths and initial-state indices, not
per-action traces or initial-state content hashes. No stronger trace claim is
made. At the later independent audit the GPU driver was unavailable; evaluation
had already finished on GPU with exit0, and no CPU rollout fallback was used.
Single-seed results do not establish stable efficacy or statistical significance.
