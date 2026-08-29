# Candidate Directional Guard Intervention Smoke

Verdict: `CANDIDATE_DIRECTIONAL_GUARD_SMOKE_SUPPORTED`
Flags: `NONE`
Branch: `research/candidate-directional-guard-smoke`
Starting SHA: `d6cdf1bfbe3a24b0c614075a83cf83a7d0736847`
Analysis code SHA at report generation: `0f90971834907cfb6e30a356ce72381222e4f7e9`
Final SHA: branch HEAD reported in the final handoff after committing this report.

## Protocol

Temporal V1; seeds 42/43; stages 8/9; previous-stage ep95 entry proxy; 10 common warmup epochs; exact paired Control/Guard branch restore; 300 intervention updates per arm; H100 primary and H300 secondary. The fixed guard scale is 0.5. Thresholds are prior H13 full/H100/endpoint_harmful/actual_step_cosine valid LOSO thresholds: seed42=0.002288496925602411 and seed43=0.004145668183330901. Old label probes use 256 records/task, fresh old score gradients use 64 records/task at every step, and the fixed current probe uses 256 records.

Branch start parity: `PASS` (maximum absolute probe difference 0).
All required metrics finite: `True`.

## H100 paired results

| Seed | Stage | Fires | Fire rate | Control old drift | Guard old drift | Old benefit | Relative old benefit | Control current drift | Guard current drift | Relative current harm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 27 | 0.27 | -0.16999425 | -0.18651378 | 0.016519534 | 0.00064795947 | -0.048475486 | -0.040010708 | 0.00048355468 |
| 42 | 9 | 20 | 0.2 | -0.1659282 | -0.17034982 | 0.0044216126 | 0.00017694703 | 0.013782781 | 0.025983124 | 0.00072458481 |
| 43 | 8 | 42 | 0.42 | 0.021075247 | -0.039419112 | 0.060494359 | 0.0023761301 | -0.041651651 | -0.0096655365 | 0.0018623944 |
| 43 | 9 | 35 | 0.35 | -0.029329636 | -0.06144658 | 0.032116945 | 0.0012748304 | -0.11148806 | -0.09743328 | 0.00082950573 |

Positive retention pairs: `4/4`; mean/median relative old benefit: `0.0011189667` / `0.00096139493`; mean/max relative current harm: `0.0009750099` / `0.0018623944`; pooled fire rate: `0.31`.

## H300 persistence

| Seed | Stage | Fires | Fire rate | Old benefit | Relative old benefit | Current harm | Relative current harm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 131 | 0.43666667 | -0.011058219 | -0.00043374576 | 0.064750155 | 0.003698885 |
| 42 | 9 | 103 | 0.34333333 | 0.012280821 | 0.00049146203 | 0.06145645 | 0.003649931 |
| 43 | 8 | 113 | 0.37666667 | 0.052733241 | 0.0020712847 | 0.06384469 | 0.0037173628 |
| 43 | 9 | 92 | 0.30666667 | 0.037480795 | 0.0014877398 | 0.037096603 | 0.0021894215 |

Positive retention pairs: `3/4`; mean relative old benefit: `0.00090418519`; mean relative current harm: `0.0033139001`; pooled fire rate: `0.36583333`.

## Per-task and mechanism observations

Task-specific retention tradeoff: `NONE`; worst task-specific relative extra harm at H100: `0.00014988014`.
Candidate cosine over all instrumented steps: mean `-0.0025116151`, median `-0.0021635651`, q25 `-0.012070278`, q75 `0.0074358231`.
Fired Guard-step cosine: mean `0.012367624`, median `0.010599335`. Mean candidate first-order score: `-0.0011568867`; summed predicted first-order score reduction over fired steps: `1.2304034`.
Intervention wall time: Control `94.392573` seconds; Guard `95.002665` seconds. Both arms recomputed fresh old gradients.

## Scientific conclusion

H15 is supported under the preregistered smoke criteria: scaling high-risk candidate displacement showed paired short-horizon retention benefit without material current-probe harm, and the mean benefit did not reverse at H300. The next action is Guard Threshold / Refresh Strategy Diagnosis; do not start formal guard training.

## Scope limits

This is a short paired BC-loss intervention smoke, not lifelong rollout performance evidence. Seed-specific LOSO thresholds are diagnostic calibration values, not a final deployable universal threshold. The intervention keeps AdamW moments on the full-gradient trajectory while scaling only parameter displacement; this is an experimental control semantics, not yet the final optimizer design. Fresh old-task gradients are recomputed every step, so runtime cost is intentionally not solved here. The old-task probe may not perfectly represent rollout retention. Ep95 remains a late-stage proxy.

## Exact successful commands

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_candidate_directional_guard.sh smoke
CUDA_VISIBLE_DEVICES=1 bash scripts/run_candidate_directional_guard.sh formal
python -m pytest -q tests/test_candidate_directional_guard.py
```
