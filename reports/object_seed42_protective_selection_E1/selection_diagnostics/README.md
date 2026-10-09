# Read-only selection preference and short-window repetition audit

This follow-up analyzes the completed OBJECT E1's existing selection logs from
input publication `e963d8d0f3a4626365d33bdb5c83a60821fff115`. It introduces no
training, rollout, model loading, random simulation or hyperparameter change.

Across 181 refresh events, 389 task-specific candidate pools and 230,710 priority
assignments, scores produced a measurable but moderate retrieval preference:

- The highest-score quarter received **37.44%** of draws (uniform: 25%; configured
  probability prediction: 37.37%). The lowest-score quarter received 16.03%.
- Mean candidate score under uniform sampling was 0.1313; actual selected score
  was 0.2558 (configured prediction: 0.2560).
- Draw-weighted mean effective sample size was **56.96 out of 64**. The candidate
  distribution did not collapse onto a few IDs.

Priority-branch repetition within each refresh window was substantial:

| Window after refresh | Observed repeat fraction | Uniform within same 64 IDs, analytic | Uniform within full old-task buffer, analytic |
|---|---:|---:|---:|
| First 50 optimizer steps | 37.42% | 35.46% | 3.88% |
| First 100 optimizer steps | 56.70% | 55.06% | 7.57% |
| Full 500 optimizer steps | 89.32% | 89.29% | 31.01% |

Repeat fraction is `(draws - distinct IDs) / draws`, resetting for every
stage/task/refresh window. Both analytic references sample **with replacement**
at the observed draw counts; neither is a newly run experiment or an observed
E0 trajectory. E0's concatenated shuffled loader is not this independent-draw
reference.

Pairwise same-ID probability separates two sources of concentration: restricting
support to 64 IDs raised the analytic collision probability about **12.01×**
relative to the full-buffer reference; score weighting increased it by another
**12.6%**. Adjacent refreshed pools overlapped by 5.36 IDs on average, close to
the random-redraw expectation of 5.35. Concentration occurs within refresh
windows, rather than through globally fixed candidate identities.

## Files and reproduction scope

- [Full Chinese report](REPORT.txt), including every stage/old-task group.
- [Structured diagnostics](SELECTION_DIAGNOSTICS.json).
- [389 per-pool rows](per_pool.csv) and [383 refresh transitions](refresh_overlap.csv).
- [Read-only analysis script](analyze_selection_logs.py) and [report renderer](write_report.py).
- [Input hashes before/after](INPUT_HASHES.json), [exact artifact hashes](OUTPUT_MANIFEST.json)
  and [input publication receipt](provenance/INPUT_PUBLICATION_RECEIPT.json).

The script uses Python and NumPy. It preserves the original execution layout:
`execution/20261009_E1_selection_diagnostics/analyze_selection_logs.py` reads
its sibling `20261008_E1_protective_selection/final_20261009/evidence/`, the
original `final_20261009/FINAL_MANIFEST.json`, and `PUBLICATION_RECEIPT.json`.
The input evidence is available in the [previous lightweight archive](../lightweight_evidence.tar.gz);
the original final manifest is [here](../FINAL_MANIFEST.json). The saved input
publication receipt must be restored as `PUBLICATION_RECEIPT.json` when reconstructing
that layout. Run the analysis in a separate output directory, not over the
completed experiment. The actual original command exited 0:

```text
python -X utf8 execution/20261009_E1_selection_diagnostics/analyze_selection_logs.py
```

All 406 relevant archived inputs matched the original final manifest, and all
analysis inputs retained identical before/after hashes.

## Interpretation limits

The logs identify priority assignments, not every kept-original sample. These
repeat fractions therefore do not describe the entire replay stream, and an
assignment may select the same ID as the sample it replaces. File IDs also do
not guarantee independent trajectories or semantic diversity.

Scores are not recomputed at each step, and component gradients are not logged.
This analysis cannot establish score staleness, useful full-network gradients,
closed-loop protection, or that candidate-pool concentration caused the observed
AUC decrease. No next experiment or parameter change follows automatically.
