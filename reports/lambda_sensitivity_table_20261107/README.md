# Retention-loss coefficient sensitivity table

Publication table matching the supplied reference: grouped LIBERO-OBJECT and LIBERO-GOAL rows, three coefficients per suite, no colored rows, three horizontal rules plus the group separator. The coefficient replaces the example's Buffer column; none of the reference image's numeric results are reused.

## Files

- `lambda_sensitivity.tex`: standalone compilable LaTeX.
- `lambda_sensitivity.pdf`: compiled one-page PDF.
- `table_lambda_sensitivity.tex`: table fragment for a manuscript. Requires booktabs, multirow and the result/bestresult macros in the standalone preamble.
- `lambda_sensitivity_preview.png`: high-resolution rendered preview.
- `per_seed_metrics.csv`, `table_values.csv`: exact numeric data.
- `TABLE_AUDIT.json`, `COMPILE_AUDIT.json`: provenance and verification.

FWT and AUC are maximized; NBT is minimized, including negative values. Each cell reports mean and sample standard deviation (ddof=1) across seeds 41, 42 and 43, rounded to one decimal. Bold is selected using exact unrounded means independently for each metric within each suite, not by coefficient row. In GOAL, the best NBT is at 0.2; it is not assigned to 0.1 merely because 0.1 has the best FWT and AUC.

## Data and verification

The builder verifies both suites' acceptance/comparison/audit hashes, recomputes 180 task-stage cells from 3600 unique accepted episode outcomes, then recomputes all 18 seed-run metric triples. It checks these against accepted case metrics and the existing cross-suite summary. The reused 0.1 baseline is checked identical in both paired comparisons and counted only once per suite. All data come from completed accepted experiments; no new training, evaluations or rollouts are performed.

FWT is the average learned diagonal; NBT averages each old task's subsequent drop from its learned diagonal; AUC averages each task over its learned stages and then averages tasks. These follow the accepted study definitions. Mean and sample SD are retained at full precision in the CSV files.

## Interpretation context

This is fixed-grid sensitivity at 0.05, 0.1 and 0.2, not a lambda=0 mechanism-removal experiment. The existing default 0.1 has the highest mean FWT and AUC in both suites and the lowest OBJECT NBT; GOAL 0.2 has the lowest NBT. The evidence does not establish a universal optimum or statistical significance. Historical 0.1 and new variants differ in hardware history; development-set reuse and unresolved pretraining provenance remain as documented in accepted reports. No parameter is selected for new runs from these results.

## Build and publication

Run `build_table.py`, compile `lambda_sensitivity.tex` with Tectonic or a compatible LaTeX engine, and run `verify_pdf.py` followed by visual inspection. The reviewed package is published to the standing authorized MLR branch without models, checkpoints or private payloads.
