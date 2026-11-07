# Main-text figures: OBJECT and GOAL

Three redesigned figures from pages 1, 3 and 5 of the original `all_figures.pdf`. All numeric evidence is unchanged. The standalone images are intended for full-width use in a two-column manuscript.

| Figure | Content | PDF | 600 dpi PNG |
|---|---|---|---|
| 1 | All-seen and old-task performance, both suites | [PDF](figures/fig01_continual_performance.pdf) | [PNG](figures/fig01_continual_performance.png) |
| 2 | Candidate tradeoffs and gate-condition patterns, both suites | [PDF](figures/fig02_gate_tradeoffs.pdf) | [PNG](figures/fig02_gate_tradeoffs.png) |
| 3 | Task-stage gains and losses, both suites | [PDF](figures/fig03_task_level_gains.pdf) | [PNG](figures/fig03_task_level_gains.png) |

[Combined three-page PDF](figures/main_text_figures.pdf) · [English captions](CAPTIONS.md)

## Changes for the main text

- The original success curves are condensed from six to four panels: all seen and old tasks, retaining both suites.
- The original gate analysis is condensed from six to four panels: candidate scatter and mutually exclusive gate-condition patterns. Stage acceptance curves remain in the original supplementary material.
- All four difference heatmaps are retained, including every negative and zero difference.
- Visible internal method codes are replaced by formal names. The complete method is Ours. The matched-optimizer baseline is explicitly labeled. The gate scatter uses candidate-specific labels to distinguish local branches from independent method runs.
- The figures use consistent Arial typography, embedded TrueType fonts, colorblind-friendly method colors with distinct markers, and common axis or color scales within each comparison.
- All panels are 7.2 inches wide as a figure, approximately 183 mm. PNGs are 4320 pixels wide at 600 dpi; heights are 3150, 3300 and 3480 pixels, respectively.
- PDFs retain vector text, axes, lines, bars and heatmap annotations. The dense scatter layers alone are rasterized at 600 dpi; every decision block is rendered without sampling or outlier removal. Editable SVGs are also retained.

## Verification

The build verifies the original source hashes and recomputes task-stage success from accepted paired outcomes. It recomputes all 47,400 gate decisions and relative-error changes from original scalar records. Reconstructed success-curve, stage-task, difference and gate-pattern tables match the original package exactly. All source evidence remains read-only. No model, checkpoint, demonstration or latent tensor is loaded; no optimizer update or environment episode is run.

`SOURCE_MANIFEST.json` binds the inputs. `DATA_AUDIT.json` records numeric and layout checks. `EXPORT_AUDIT.json` verifies font embedding, absence of Type3 fonts, figure dimensions, PNG resolution and PDF text, and binds final PDF hashes. `VISUAL_REVIEW.json` records inspection of the final PDF renders. The numeric CSV files preserve source identifiers internally for traceability; internal codes are not visible in the figures.

## Rebuild

On the original Windows workspace, use Python 3.13 with `tmp/publication_plot_deps` on PYTHONPATH to run `make_figures.py`. That local plotting dependency directory is separate from scientific environments. Run `assemble_and_check.py` with the bundled Python runtime containing pypdf, pypdfium2 and Pillow, then inspect all three PDF renders before publishing. Publication includes only reviewed code, captions, scalar tables and figure assets.

The broader hardware, training-budget and development-protocol context remains in the accepted study reports. These figures describe existing results; no additional significance or single-mechanism causal claim is introduced by the redesign.
