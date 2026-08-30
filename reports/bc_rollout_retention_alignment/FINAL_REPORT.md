# BC-Loss–Rollout Retention Alignment Diagnosis

## Identity

- Verdict: `BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_SUPPORTED`
- Flags: `LONGITUDINAL_BC_ROLLOUT_ALIGNMENT_WEAK`
- Branch: `research/bc-rollout-retention-alignment`
- Starting SHA: `80dfb5a5b25cf2b7ed18ed530b8291703c8e3cdf`
- Analysis source SHA: `4e3448c09b44c00c5b190ab49cc5138c5a9b1e4a`
- Final SHA: recorded by the result commit and final handoff
- Push status: `PUSHED`

## Protocol

No retraining was performed. The analysis compares Temporal V1 and Zero Guard
for seeds 42/43/44 at epochs 70,75,80,85,90,95. BC loss uses the frozen
256-record/task manifest with batch size 32. Rollout success uses 20 trials per
checkpoint-task and the existing formal LIBERO_OBJECT evaluation path.

## Primary alignment

| Metric | Value |
|---|---:|
| Matched rows | 108 |
| Rollout non-tie rows | 87 |
| Pearson | -0.018820 |
| Spearman | -0.038014 |
| ROC-AUC, non-tie | 0.462073 |
| Average precision, non-tie | 0.519738 |
| Sign concordance, non-tie | 0.505747 |
| BB / BW / WB / WW / ties | 30 / 25 / 18 / 14 / 21 |
| Dangerous BW rate | 0.454545 |

## Per seed

| Seed | Spearman | AUC | Concordance | BW rate |
|---:|---:|---:|---:|---:|
| 42 | -0.030214 | 0.462963 | 0.400000 | 0.625000 |
| 43 | 0.139753 | 0.608392 | 0.625000 | 0.428571 |
| 44 | 0.196504 | 0.525000 | 0.515152 | 0.235294 |

## Per stage

| Stage | Spearman | AUC | Concordance | BW rate |
|---:|---:|---:|---:|---:|
| 7 | -0.102195 | 0.425926 | 0.466667 | 0.444444 |
| 8 | -0.082320 | 0.395722 | 0.500000 | 0.411765 |
| 9 | -0.034822 | 0.500000 | 0.522727 | 0.482759 |

## Longitudinal

| Method | Valid relations | Median Spearman | Adjacent concordance | BC-improve / rollout-worse |
|---|---:|---:|---:|---:|
| temporal_v1 | 18 | -0.168479 | 0.480519 | 17 |
| zero_guard | 18 | 0.366130 | 0.592593 | 16 |


## NBT surrogate

The analysis contains 18 method-task rows and
9 matched Guard-vs-V1 task contrasts. Non-tie sign
concordance is 0.555556; BC-better/rollout-worse occurs in
1 contrasts.

## Observed

The complete checkpoint grids, matched quadrants, longitudinal transitions,
selected epochs, and NBT analogues are retained in the CSV files in this
directory. The evidence floor was met.

## Inferred

H18 is classified as `BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_SUPPORTED` using the preregistered thresholds.
Mean old-task BC loss is not sufficiently aligned with rollout retention to be the sole control surrogate for the current Guard.


## Not established

Weak alignment does not prove BC loss is useless for imitation learning.

This concerns mean BC loss on fixed probes, not all possible state-conditioned
or trajectory-conditioned behavioral losses.

20-trial rollout has 5-pp resolution and can create ties.

Correlation/mismatch does not identify a unique causal mechanism.

Analysis is limited to LIBERO_OBJECT and ep70-95 checkpoints.

No new training was performed.
