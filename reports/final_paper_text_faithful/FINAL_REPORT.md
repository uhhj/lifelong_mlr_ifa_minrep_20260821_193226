# Final paper-text-faithful reproduction

## Verdict

`PAPER_TEXT_FAITHFUL_METHOD_LEVEL_REPRODUCTION_NOT_SUPPORTED`

Flags: `CHECKPOINT_SELECTION_MATERIAL`; structural collapse: `False`.

Provenance verdict: `PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE`.

## Paper-explicit protocol

LIBERO-OBJECT; batch 10; 100 epochs per lifelong stage; AdamW at initial LR 1e-4; linear scheduler; replay equivalent to five demonstrations per task; IFA lambda 0.1 and alpha/gamma 0.30; seeds 42/43/44; 20 evaluation trials.

## Paper-unspecified frozen choices

Scheduler advancement after every optimizer step (**PAPER DOES NOT SPECIFY SCHEDULER STEP UNIT**); warmup 500 (**RELEASE_FALLBACK**); AdamW betas [0.9, 0.999], eps 1e-8 and weight decay 0.1 (**RELEASE_FALLBACK**); existing seed-specific augmented feature caches (**FEATURE_AUGMENTATION = RELEASE_FALLBACK; NOT PAPER-SPECIFIED**); release GMM stochastic action sampling (**RELEASE_IMPLEMENTATION_FALLBACK**); ep99 primary checkpoint; release-compatible selected-checkpoint secondary view.

## Pretraining limitation

This is a **LIFELONG-STAGE PAPER-TEXT-FAITHFUL REPRODUCTION WITH RELEASED/FROZEN PRETRAINED INITIALIZATION**. Exact paper-time pretraining code is not public; every run uses the same frozen `object_multitask_model_ep99.pth`. This is not a full end-to-end exact-paper reproduction.

## Environment

- Python: 3.8.20
- GPU: NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090
- PyTorch: 1.13.0+cu116; MuJoCo: 3.2.3; robosuite: 1.4.0; transformers: 4.26.1; CLIP: not-distribution-metadata

## Primary ep99 results

- MLR: FWT 82.500 ± 6.495 / NBT 3.889 ± 5.300 / AUC 80.104 ± 8.006
- MLR+IFA alpha0.30: FWT 76.667 ± 6.292 / NBT 5.185 ± 6.783 / AUC 74.028 ± 9.214
- Mean IFA - MLR: ΔFWT -5.833; ΔNBT +1.296; ΔAUC -6.076 pp.
- Per-seed signs (42/43/44): FWT ['+', '-', '-']; NBT ['-', '+', '+']; AUC ['+', '-', '-'].

| Method | Seed | FWT | NBT | AUC |
| --- | ---: | ---: | ---: | ---: |
| MLR | 42 | 75.000 | 7.222 | 71.146 |
| MLR | 43 | 86.250 | 6.667 | 82.604 |
| MLR | 44 | 86.250 | -2.222 | 86.562 |
| IFA | 42 | 77.500 | -0.278 | 77.708 |
| IFA | 43 | 70.000 | 12.778 | 63.542 |
| IFA | 44 | 82.500 | 3.056 | 80.833 |

## Secondary checkpoint sensitivity

- MLR: FWT 90.417 ± 2.602 / NBT 12.963 ± 11.185 / AUC 83.715 ± 7.382
- MLR+IFA: FWT 86.667 ± 1.909 / NBT 13.241 ± 9.955 / AUC 79.965 ± 6.510
- Mean IFA - MLR: ΔFWT -3.750; ΔNBT +0.278; ΔAUC -3.750 pp.
- Flag: `CHECKPOINT_SELECTION_MATERIAL`. The flag uses the frozen rule `|secondary-primary FWT| >= 2pp OR |secondary-primary AUC| >= 2pp` for either method.
- Selected epochs are in `secondary_selected/selected_epochs.csv`.

| Method | Seed | FWT | NBT | AUC | Selected epochs for stages 6/7/8/9 |
| --- | ---: | ---: | ---: | ---: | --- |
| MLR | 42 | 87.500 | 23.333 | 75.938 | [85, 99, 75, 95] |
| MLR | 43 | 92.500 | 14.444 | 84.583 | [80, 99, 95, 80] |
| MLR | 44 | 91.250 | 1.111 | 90.625 | [90, 90, 85, 99] |
| IFA | 42 | 86.250 | 11.667 | 80.000 | [85, 99, 95, 70] |
| IFA | 43 | 85.000 | 23.889 | 73.438 | [80, 99, 80, 75] |
| IFA | 44 | 88.750 | 4.167 | 86.458 | [90, 99, 99, 80] |

## Paper comparison

Paper MLR: FWT 83.3±2.6 / NBT 12.3±2.4 / AUC 77.6±3.0. Paper IFA: FWT 84.6±1.9 / NBT 11.4±5.6 / AUC 79.4±1.5. Exact signed gaps and local sample standard deviations are in `comparison/paper_vs_primary.csv` and `comparison/paper_vs_secondary.csv`; performance above the paper mean is not treated as failure.

Primary signed gaps (ours minus paper):

- MLR FWT: -0.800 pp
- MLR NBT: -8.411 pp
- MLR AUC: +2.504 pp
- IFA FWT: -7.933 pp
- IFA NBT: -6.215 pp
- IFA AUC: -5.372 pp

## Per-stage interpretation

Largest primary current-task method difference is stage/task 9: MLR mean 86.667, IFA mean 58.333, IFA-MLR -28.333 pp. Structural collapse rule: same current task at or below 10 pp in at least two of three seeds for either method. Detected items: `[]`.

## Final interpretation

What reproduced: the frozen paper-explicit lifelong configuration, paired three-seed MLR/IFA execution, primary final-state evaluation, secondary checkpoint sensitivity, and method-level comparison. What remains unrecoverable: the exact paper-time pretraining implementation and all paper-unspecified choices listed above. These results are the final paper-text-faithful local baseline; no further batch/gamma/scheduler/augmentation/checkpoint search will be used to chase Table 1.

Branch: `research/final-paper-text-faithful-reproduction`. Starting SHA: `1f4e45d604be7f2b61b356e70ac4fa7c80557fe0`. The exact final commit SHA is recorded in the final Git handoff because a commit cannot contain its own hash.

## Exact successful commands

```bash
bash scripts/run_final_paper_text_faithful.sh protocol
bash scripts/run_final_paper_text_faithful.sh test
bash scripts/run_final_paper_text_faithful.sh smoke
bash scripts/run_final_paper_text_faithful.sh formal
```
