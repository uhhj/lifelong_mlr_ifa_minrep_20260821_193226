# OBJECT / GOAL — publication figure package

Four main figure groups and two supplementary groups, generated solely from accepted existing evidence. All groups cover both OBJECT and GOAL. English plot labels and captions support direct manuscript insertion; Chinese notes explain interpretation and reporting conventions.

|Figure|Purpose|PDF|PNG (600dpi)|SVG|
|---|---|---|---|---|
|1|Success curves: all seen / old / new tasks|[PDF](figures/fig01_success_curves.pdf)|[PNG](figures/fig01_success_curves.png)|[SVG](figures/fig01_success_curves.svg)|
|2|C/Q/S task-by-stage matrices|[PDF](figures/fig02_task_heatmaps.pdf)|[PNG](figures/fig02_task_heatmaps.png)|[SVG](figures/fig02_task_heatmaps.svg)|
|3|All gate candidates, condition patterns, stage acceptance|[PDF](figures/fig03_gate_tradeoffs.pdf)|[PNG](figures/fig03_gate_tradeoffs.png)|[SVG](figures/fig03_gate_tradeoffs.svg)|
|4|Three-lambda AUC / final all / final old / final new|[PDF](figures/fig04_lambda_sensitivity.pdf)|[PNG](figures/fig04_lambda_sensitivity.png)|[SVG](figures/fig04_lambda_sensitivity.svg)|
|S1|S−C and S−Q task-level differences|[PDF](figures/figS01_task_differences.pdf)|[PNG](figures/figS01_task_differences.png)|[SVG](figures/figS01_task_differences.svg)|
|S2|Three-lambda FWT / NBT|[PDF](figures/figS02_lambda_fwt_nbt.pdf)|[PNG](figures/figS02_lambda_fwt_nbt.png)|[SVG](figures/figS02_lambda_fwt_nbt.svg)|

[Six-page combined PDF](figures/all_figures.pdf) · [English captions and Chinese notes](CAPTIONS.md)

## Figure design

- Sized for double-column use (mostly 7.5–7.7 inches wide), Arial 7–9pt labels, embedded TrueType PDF fonts, no Type3 fonts.
- Consistent C gray / Q orange / S teal; marker shapes distinguish methods in monochrome. Lambda panels use marker shape for seed identity and black diamonds for means.
- Absolute heatmaps share 0–100% limits; difference heatmaps share symmetric zero-centered limits; unobserved tasks are masked.
- Curves show means ± sample SD across three seeds. Lambda figures display every seed rather than a pooled episode confidence interval.
- Gate plots include every block, with no downsampling or deleted outliers. Only dense scatter layers are rasterized (600dpi); other PDF elements are vector. SVG text remains editable and requires Arial or a deliberate font substitution on another machine.
- All six sensitivity metrics are covered between Figures 4 and S2.

## Data and reproduction

`SOURCE_MANIFEST.json` binds accepted source files by SHA-256. `data/` contains full plotted scalar tables, paired outcomes, and all 47,400 gate-block deltas. `DATA_AUDIT.json` records outcome/metric/gate recomputation checks; `EXPORT_AUDIT.json` checks PDF font embedding, text extraction, page dimensions, PNG resolution, and PDF rendering; `VISUAL_REVIEW.json` records the final inspected version. `FIGURE_MANIFEST.json` hashes the exported figures.

Run `make_figures.py` with Python, NumPy, Matplotlib and Pillow; versions used are recorded in DATA_AUDIT.json. Then run `assemble_and_check.py` with pypdf, pypdfium2 and Pillow. The scripts locate source evidence relative to the existing execution directory. No models, checkpoints, demonstrations or tensor payloads are loaded. Local plotting dependencies were installed under the workspace `tmp/publication_plot_deps`, outside the publication package and scientific environments.

On this Windows workspace:

```powershell
$env:PYTHONPATH=(Resolve-Path 'tmp/publication_plot_deps').Path
python -X utf8 execution/20261103_lambda_r_goal_sensitivity/publication_figures_20261106/make_figures.py
Remove-Item Env:PYTHONPATH
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 execution/20261103_lambda_r_goal_sensitivity/publication_figures_20261106/assemble_and_check.py
```

## Method labels and scope

C is the accepted matched-optimizer-budget C (100 epochs at Task6, 200 at Task7–9); Q and S use 100 epochs per task. C/S match physical production optimizer calls; Q/S match retained updates and collection counts. Main C/Q/S plots use default lambda=0.1 S. Gate candidates named C/Q are local branches inside S, not the independently trained methods. Gate probes are training diagnostics, while success curves and matrices use formal outcomes. Source reports retain all hardware, cost and evaluation-provenance details. No new experimental runs were performed for this package.

Publication includes only reviewed plotting code, figure assets, scalar/outcome tables, captions and hashes. No model, checkpoint, cache payload, latent tensor or credentials are included.
