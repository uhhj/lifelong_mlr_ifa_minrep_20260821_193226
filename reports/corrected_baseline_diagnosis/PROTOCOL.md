# Corrected-baseline interference diagnosis

Frozen source: `d75fb371cf575beb079ce064d70d2ae323da95dd`.
Branch: `codex/corrected-baseline-interference-diagnosis`.

Question: where does corrected IFA's remaining task7 deficit occur, and what
changes in the task9 representation/BC objective are already present before task9?
No optimizer, parameter updates, new training, new rollout protocol, or tuning.
Original runs/checkpoints/features/buffers remain read-only. No algorithm changes.

## Prespecified diagnostic grid

- Arms: existing MLR, release-reference IFA, corrected-reference IFA.
- Seeds: 42,43,44. Fixed end-of-stage checkpoints: tasks6,7,8,9, epoch99.
- Tasks probed: 6–9, including future tasks strictly for offline diagnosis.
- Five equal trajectory-progress bins; two deterministic batches of10 per bin/task.
  Same examples for every checkpoint and arm within a seed. Diagnostic sampling
  seed is20260906 plus explicit seed/task/bin offsets. No selection by loss/result.
- Reuse existing bin metadata, but independently verify every selected feature's
  entire action sequence against first10 HDF5 demos with end padding. Require one
  unique match, and agreement with its progress bin. Save demo and sequence index.
- Dropout off (`eval`) for reproducible frozen-checkpoint geometry; GMM BC uses
  learned variance, no action sampling. This is not a historical training-gradient
  reconstruction or rollout evaluation. Training behavior is not changed.
- Full trainable decoder+head gradients. Report same-bin task-pair BC cosine and
  corrected IFA-vs-current/old/future-task BC cosine and norm. No AdamW state exists
  in these probes, so do not interpret these as actual optimizer-step directions.
- Corrected IFA formula is the existing twice-normalized angular hinge with the
  frozen config's gamma and weight. Recover pairs with the official pair builder.
  IFA probes outside the pair's actual stage are explicitly counterfactual probes,
  not assertions that IFA was active there. Task9 has no direct IFA term.

## Checks and interpretation

Require strict checkpoint loading, finite losses/gradients, unchanged model tensors,
all36 checkpoints,1440 BC rows,2160 BC-pair rows and1920 IFA-pair rows. All arm/seed
task6 checkpoints must yield exactly equal probe results within a seed.
Each two-batch bin estimate is exploratory and demo samples are correlated. Raw
probe rows are not independent seed replicates; no significance tests. Progress
bins are not semantic behavior labels. BC/gradient evidence cannot by itself
establish success-critical retention or causal rollout effects.

Report paired task7/task9 BC profiles, checkpoint-stage changes, gradient sign
patterns, and link them descriptively to the already committed success matrices.
No new method is launched based on this diagnostic alone.
