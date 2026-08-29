# Online First-Order Margin Validity Diagnosis

## Identity

- Verdict: `ONLINE_FIRST_ORDER_MARGIN_VALIDITY_INCONCLUSIVE`
- Attribution: `ONLINE_MARGIN_SIGNAL_NOT_ESTABLISHED`
- Branch: `research/online-first-order-margin-validity`
- Starting SHA: `d725842ed792e40b30d1965a395315829c29256b`

## Protocol

Temporal V1 targeted continuations used seeds 42/43/44, stages 8/9, 15 epochs, batch size 10, replay rate 0.5, five bins, gate strength 1.0, and IFA disabled. Entry checkpoints were previous-stage ep95 proxies. The first two epochs were burn-in; anchors were sampled every 100 optimizer updates. Label and score-gradient probes used 256 and 64 fixed records per old task.

## Training parity

Instrumentation parity passed before formal execution. The shadow score restores CPU/CUDA RNG and train/eval mode, while real updates call the committed `MLRTemporalRobotIGFA.observe()` directly. Optimizer and scheduler semantics match the release, including 100-epoch total-step construction and epoch-end scheduler stepping.

## Coverage

- All anchors: 267
- Anchors with native candidate bins: 261
- Anchor candidate coverage: 97.75%
- H100 native cases: 255 of 261 (97.70%)
- H100 native cases per seed: {42: 84, 43: 85, 44: 86}

## Primary native raw

- Harmful prevalence: 35.69%
- ROC-AUC / AP: 0.5771 / 0.3937
- Threshold0 BA / false-safe / false-alarm: 54.87% / 5.88% / 47.45%
- LOSO BA / false-safe / false-alarm: 56.87% / 12.94% / 32.16%
- BA gain / false-safe reduction: +2.00 pp / -7.06 pp
- Held-out seeds BA>=0.60: 1/3

| Held-out seed | Train seeds | tau | Test BA | Sensitivity | Specificity | False-safe |
|---:|---|---:|---:|---:|---:|---:|
| 42 | 43+44 | 9031.07 | 52.96% | 63.33% | 42.59% | 13.10% |
| 43 | 42+44 | 6130.7 | 54.72% | 61.29% | 48.15% | 14.12% |
| 44 | 42+43 | 12383.8 | 62.80% | 66.67% | 58.93% | 11.63% |

## Native cosine

ROC-AUC/AP 0.5586/0.3695; LOSO BA 56.14%; false-safe reduction -6.27 pp.

## Actual optimizer-step bridge

ROC-AUC/AP 0.6587/0.4733; LOSO BA 66.14%; LOSO false-safe 13.03%.

## Incremental margin analysis

- Currently ungated / harmful: 58 / 15
- Newly flagged / harmful: 0 / 0
- Incremental harmful recall: 0.00%
- Incremental false alarms: 0
- Harmful still missed: 33

## H300 secondary

- Native raw AUC/LOSO BA: 0.6110/53.66%
- Native cosine AUC/LOSO BA: 0.5573/57.95%
- Actual-step AUC/LOSO BA: 0.6934/68.63%
- Delayed raw signal criterion: not met

## Stage8/9

- Stage 8: native_raw_max AUC/BA=0.5844/56.48%; native_cos_max AUC/BA=0.5779/60.18%; actual_step_score AUC/BA=0.6382/65.21%.
- Stage 9: native_raw_max AUC/BA=0.5695/57.40%; native_cos_max AUC/BA=0.5438/53.14%; actual_step_score AUC/BA=0.6797/67.06%.

## Observed

Anchor losses, native pre-gate risks, realized optimizer-step scores, future loss deltas, ranking metrics, LOSO folds, and incremental margin counts are direct outputs of the targeted continuations.

## Inferred

H11 is `ONLINE_FIRST_ORDER_MARGIN_VALIDITY_INCONCLUSIVE` with preregistered attribution `ONLINE_MARGIN_SIGNAL_NOT_ESTABLISHED`.

## Not established

Ep95 is a late-stage entry proxy, not the exact ep99 boundary. Continuations are distribution-matched, not exact historical minibatch replay. LOSO over three seeds is descriptive rather than a significance claim. BC-loss retention is not rollout-success retention, and diagnostic tau is not yet a validated formal training hyperparameter.

## Verdict

`ONLINE_FIRST_ORDER_MARGIN_VALIDITY_INCONCLUSIVE`

## Attribution

`ONLINE_MARGIN_SIGNAL_NOT_ESTABLISHED`

## Next action

Resolve cross-seed or horizon instability before any margin intervention.
