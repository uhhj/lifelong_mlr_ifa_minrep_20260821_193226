# Optimizer-budget comparison

Separate companion to the component-ablation table. No new training, rollouts or evaluations were performed.

## Deliverables

- `optimizer_budget.tex`: standalone compilable source, matching the main table style.
- `optimizer_budget.pdf`: compiled one-page PDF.
- `table_optimizer_budget.tex`: table-only fragment for insertion into the manuscript.
- `optimizer_budget_preview.png`: rendered preview.
- `per_seed_metrics.csv` and `table_values.csv`: exact values before one-decimal display rounding.
- `TABLE_AUDIT.json` and `COMPILE_AUDIT.json`: source hashes, recomputation and layout checks.

The fragment requires booktabs, multirow, graphicx and xcolor with the table option, plus the `oursblue`, `result` and `bestresult` definitions in the standalone source.

## Configurations and budget

Execution Supervision is RA-Coreset + ALIGN with execution supervision (the ordinary-budget baseline in the component table). The matched-budget row extends this same method; it does not add a mechanism. Ours is the complete method, reusing accepted results.

| Configuration | Epochs per stage (Tasks 6/7/8/9) | OBJECT optimizer calls/run | GOAL optimizer calls/run |
|---|---|---:|---:|
| Execution Supervision | 100/100/100/100 | 105500 | 76100 |
| + Matched Optimizer Budget | 100/200/200/200 | 195300 | 142200 |
| Ours | 100/100/100/100 | 195300 | 142200 |

Counts are physical production optimizer calls, including discarded candidates, excluding separately accounted admission diagnostics. Stage-wise equality for matched baseline versus Ours is checked for all six suite/seed pairs against accepted comparison evidence. The baseline learning-rate schedule is extended to its actual update horizon with the same warmup. Ordinary-budget counts follow the accepted 100-epoch stage budgets. This matches optimizer calls, not FLOPs, wall-clock time or training interactions: Ours has 40 historical training collection episodes per run and the execution-supervision baselines have none.

## Recalculation

All 18 seed-level matrices are recomputed from the accepted local summary. Seeds are 41, 42 and 43; standard deviations use ddof=1. FWT averages diagonal success; NBT averages each old task's post-learning drop relative to its learned diagonal, then averages old tasks; AUC averages each task's success over the stages when it has been learned, then averages tasks. Best means use exact unrounded values, with lower NBT better. Sources and hashes are recorded in TABLE_AUDIT.json. Implementation and protocol provenance remain in the existing lineage audit.

## Reading the result

At the matched optimizer budget, Ours has higher mean FWT and AUC and lower mean NBT than the extended baseline on both suites. Relative to the ordinary baseline, extended training changes OBJECT AUC from 85.0000 to 80.9028 and GOAL AUC from 82.5000 to 83.7153; Ours reaches 86.4931 and 87.2569. More optimizer updates therefore do not remove the observed mean advantage of the complete method in these seeds. The table is a budget control, not proof of statistical significance or single-mechanism causality. Ordinary execution supervision still has the highest OBJECT mean FWT among these rows.

## Rebuild

Run `build_table.py`, compile `optimizer_budget.tex` with Tectonic or a compatible LaTeX compiler, then run `verify_pdf.py` and visually inspect its rendered preview. The audited PDF was compiled with Tectonic 0.17.0. Publication uses the standing authorized branch and only this reviewed lightweight package.
