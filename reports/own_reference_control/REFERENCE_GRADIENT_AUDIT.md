# Read-only IFA reference gradient diagnostic

Formal source commit: `a4ed54d8502a10dd41a1a5ec83f9234e97a0071a`.

Executed on the lab server, GPU1, Python3.8/PyTorch1.13.0+cu116. Script: `audit_reference_gradient.py`; raw results: `reference_gradient_result.json`.

## Scope

54 diagnostic batches: seeds42/43/44 stage7 and seed42 stage8, checkpoints ep70 and ep99. For each checkpoint the first three eligible batches per first-sample source category were selected from that epoch's deterministic sampler order. The existing END-of-epoch model was evaluated, not the historical within-epoch model state. This is a stratified diagnostic sample, not an unbiased estimate of training-wide loss or gradient averages.

Formal checkpoint weights were loaded with strict=True into BriefPolicy. Original frozen feature caches and official similarity/margin construction were used. Final replay was filtered to tasks before each stage; the prior sampling audit established append-only replay and exact final filename equality. A single train-mode forward with fixed diagnostic RNG was shared by both reference computations, eliminating between-arm dropout differences. Gamma0.30, lambda0.1, double l2norm and acos clamping matched release operations. Only the own-reference lookup changed in the diagnostic expression.

The probe used `autograd.grad` with respect to detached current-task global latent g. No parameter gradient was computed; no optimizer was created or stepped. No source, model, feature, buffer, or formal result was changed. Diagnostics were run from /tmp, with output captured locally.

## Results

All 54 pairs of losses and gradients were finite.

| Category | Batch count | Observation |
| --- | ---: | --- |
| First sample belongs to current task | 24 | Own-reference vectors identical; losses and gradients identical; gradient cosine approximately1 |
| First sample belongs to selected negative task | 24 | Release positive and negative reference vectors exactly identical; release gradient with respect to g exactly zero in all24; corrected gradient nonzero in all24 |
| First sample belongs to another old task (seed42 stage8) | 6 | Both gradients nonzero; cosine0.33755–0.35746 |

Means below are over six sampled batches per row (three per checkpoint). Loss includes lambda0.1. Gradient norm is with respect to g, not model parameters.

| Seed/stage/category | Release loss | Corrected loss | Release gradient norm | Corrected gradient norm |
| --- | ---: | ---: | ---: | ---: |
| 42/7/negative | .021341624 | .008937651 | 0 | .001816044 |
| 43/7/negative | .021341625 | .008487915 | 0 | .002093113 |
| 44/7/negative | .021341626 | .009333956 | 0 | .001893206 |
| 42/8/negative | .027616201 | .009468389 | 0 | .003665850 |
| 42/8/other old | .025569329 | .010940728 | .002490093 | .003192918 |

## Meaning and limits

The own-reference indexing mismatch is not merely a logging or naming defect. The sampled real features and checkpoints show either cancellation or alteration of the IFA representation gradient. When positive and negative references are identical, the two angular terms cancel and leave a positive fixed margin: nonzero IFA loss alone does not establish effective regularization. The BC term remains separate and can still update the policy.

These diagnostics do not quantify how much of the observed FWT/AUC gap the defect caused and do not establish that a corrected three-seed experiment would recover the paper's gains. They establish that the executed release objective does not consistently implement the paper's own-task-reference mechanism.

Preserve original numerical results as release-implementation evidence. A subsequent correction and controlled re-evaluation would be a separate experiment, requiring explicit scope; no retraining was launched here.
