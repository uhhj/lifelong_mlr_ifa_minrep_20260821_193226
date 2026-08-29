# Candidate AdamW-Step Cosine Feasibility Diagnosis

## Identity

- Verdict: `CANDIDATE_ADAMW_STEP_COSINE_FEASIBILITY_SUPPORTED`
- Branch: `research/candidate-adamw-step-cosine-feasibility`
- Starting SHA: `25065cdeed8e4e6fe7ebf7b9b5c20a4419ffcc90`
- Final SHA: branch HEAD containing this report (reported in the external handoff)

## Protocol

Temporal V1 targeted continuations used seeds 42/43/44, stages 8/9, previous-stage ep95 entry proxies, 15 epochs, batch size 10, replay rate 0.5, five temporal bins, gate strength 1.0, and IFA disabled. Anchors begin after two burn-in epochs and occur every 100 optimizer updates. H100 is primary and H300 secondary.

Formal completeness: met; anchors=267, H100=261, H300=249.

## Optimizer semantics

| Optimizer | Groups | Configured LR | Betas | Eps | Weight decay | AMSGrad | Maximize | Grad clipping |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| AdamW | 1 | 0.0001 | 0.9,0.999 | 1e-08 | 0.1 | 0 | 0 | 100.0 |

## Candidate formula

The candidate uses the current post-clip gradient and current AdamW moments/step, bias-corrected first and second moments, and decoupled weight decay. It mirrors Torch 1.13's decay-then-addcdiv parameter operations on a private clone and does not mutate parameters, gradients, or optimizer state.

## Numerical displacement parity

- Candidate-actual cosine mean/median/min: 1.000000000 / 1.000000000 / 1.000000000
- Relative L2 median/p99/max: 0 / 0 / 0
- Maximum absolute tensor delta error: 0

## Risk cosine parity

- MAE/p99/max: 0 / 0 / 0
- Pearson/Spearman: 1.000000000 / 1.000000000
- Sign agreement: 100.00%

## H100

| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |
|---|---:|---:|---:|---:|---:|
| candidate cosine | 0.7153 | 0.5425 | 66.73% | 63.04% | 13.03% |
| actual cosine | 0.7153 | 0.5425 | 66.73% | 63.04% | 13.03% |

Candidate-actual AUC delta +0.0000; LOSO BA delta +0.00 pp.

## H300

| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |
|---|---:|---:|---:|---:|---:|
| candidate cosine | 0.7393 | 0.5550 | 69.19% | 65.00% | 11.24% |
| actual cosine | 0.7393 | 0.5550 | 69.19% | 65.00% | 11.24% |

## Seed-stage

| Seed | Stage | Anchors | Median delta cosine | p99 relative L2 | Risk cosine MAE | H100 candidate AUC | H100 actual AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 41 | 1.000000000 | 0 | 0 | 0.6693 | 0.6693 |
| 42 | 9 | 48 | 1.000000000 | 0 | 0 | 0.7646 | 0.7646 |
| 43 | 8 | 41 | 1.000000000 | 0 | 0 | 0.7413 | 0.7413 |
| 43 | 9 | 48 | 1.000000000 | 0 | 0 | 0.7098 | 0.7098 |
| 44 | 8 | 41 | 1.000000000 | 0 | 0 | 0.7053 | 0.7053 |
| 44 | 9 | 48 | 1.000000000 | 0 | 0 | 0.7406 | 0.7406 |

## Observed

Candidate/actual displacement parity, aggregate old-gradient cosine parity, future BC-loss labels, ranking metrics, and LOSO folds are direct outputs of the six formal targeted continuations.

## Inferred

H14 is `CANDIDATE_ADAMW_STEP_COSINE_FEASIBILITY_SUPPORTED`. The pre-step AdamW candidate reconstructs the post-step directional diagnostic under the preregistered criteria.

## Not established

Candidate cosine is pre-step computable only because the current gradient and current AdamW state are available. This diagnosis does not show that rejecting or modifying a high-risk candidate improves lifelong performance. The score still uses an old-task probe gradient, so runtime cost and probe representativeness remain open issues. The targeted continuations are not rollout evaluation. BC-loss risk is not identical to rollout-success forgetting. The ep95 entry remains a late-stage proxy.

## Next action

Run a preregistered Candidate Directional Guard Intervention Smoke on seeds 42/43 and stages 8/9; do not start formal guard training.
