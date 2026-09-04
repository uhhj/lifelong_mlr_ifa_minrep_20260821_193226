# Paper Exact-Recipe Source Archaeology

## Verdict

**PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE**

## Flags

PUBLIC_HISTORY_PRE_PAPER_EXISTS, PUBLIC_HISTORY_INCOMPLETE_FOR_EXACT_RECIPE, PAPER_BATCH_RELEASE_MISMATCH_CONFIRMED, PAPER_GAMMA_RELEASE_MISMATCH_CONFIRMED, SCHEDULER_UNIT_MISMATCH_PRESENT_AT_HEAD, FEATURE_AUGMENTATION_UNDERSPECIFIED, CHECKPOINT_SELECTION_NOT_PUBLICLY_SPECIFIED, EVALUATION_PROTOCOL_PARTIALLY_SPECIFIED, NO_HISTORICAL_COMMIT_MATCHES_ALL_PAPER_TEXT, NO_PUBLIC_PAPER_TIME_SNAPSHOT

## Branch / SHA

- Starting SHA: `419ecfb366b77207e44efb73667a1e5cfc7f3040`
- Branch: `research/paper-exact-recipe-source-archaeology`
- Execution scope: static Git/paper inspection only; no training, optimizer step, feature extraction, rollout, or GPU execution.

## Public repository history

The public repository has 9 reachable commits. The oldest author date is 2026-03-09T14:24:44+01:00, the oldest commit date is 2026-03-09T14:24:44+01:00, and the newest commit date is 2026-06-15T22:48:18+02:00. Although README-only history exists before arXiv, the first public source-code snapshot is `3e5b22f3d1b23a2bf30cbe375c600110dbf302a9` dated 2026-06-03T14:50:00+02:00, after the arXiv submissions. GitHub metadata status: OK.

## Paper timeline

- arXiv v1: 2026-03-11
- arXiv v2: 2026-03-12
- CVPR publication: June 2026

## Paper textual recipe

The paper explicitly states batch 10, 100 epochs, AdamW with initial LR 1e-4 and a linear scheduler, IFA weight 0.1, OBJECT alpha 0.30, three seeds, 20 trials per seed, and a five-demonstration-equivalent latent buffer. It does not state the scheduler step unit, feature-augmentation setting, or checkpoint-selection rule.

## Current release recipe

Static parsing gives effective batch 18, 100 epochs, AdamW LR 0.0001 with weight decay 0.1, warmup 500, scheduler semantics **EPOCH_STEP_WITH_BATCH_TOTAL**, release gamma 0.31, feature extraction **FEATURE_AUG_ON**, evaluation config n_eval=2.0, checkpoints 70;75;80;85;90;95, and action semantics STOCHASTIC_GMM_SAMPLE.

## Historical recipe timeline

No source code exists in the public history until 2026-06-03. The June 15 source change only alters evaluation logging/comments and the default benchmark/checkpoint in `test.sbatch`; it does not establish a different Table 1 recipe.

## Batch history

The only public code snapshot uses adaptation-config batch 18 and provides no OBJECT launcher override. This conflicts with paper batch 10.

## Scheduler history

Every public code snapshot computes total steps as epochs × loader length but calls `scheduler.step()` once after each epoch: **EPOCH_STEP_WITH_BATCH_TOTAL**. No public batch-step snapshot exists, so the mismatch cannot be attributed to a post-paper cleanup change from public history.

## Gamma history

Every public OBJECT launcher snapshot uses config symbol `gamma=0.31`; the loss multiplies the same angular reference distance described as alpha in the paper. Thus gamma and alpha are semantically corresponding factors, and public code conflicts with paper alpha 0.30.

## Feature extraction history

Every public feature launcher sets `+aug=True`; `train.use_augmentation=true`, and preprocessing invokes stochastic image transforms. `control_seed(cfg.seed)` seeds the process, so extraction is stochastic but seed-controlled. The paper does not specify feature augmentation.

## Evaluation history

The release evaluates epochs 70,75,80,85,90,95 but exposes no official aggregation or checkpoint-selection procedure. Its default `n_eval=2`, despite paper 20 trials. Initial-state indexing is fixed, environment seeding is applied, and `CompletePolicy.get_action` samples from the GMM, so actions are stochastic. The released test loop resets the global seed once per seed before sequential checkpoint evaluation, not before every checkpoint.

## Candidate paper-time commits

The highest textual-match public candidate is `3e5b22f3d1b23a2bf30cbe375c600110dbf302a9` (2026-06-03T14:50:00+02:00, fraction 0.625), but it is only a **WEAK_CANDIDATE**: its code was published after arXiv and retains batch/gamma/evaluation conflicts. No public paper-time source candidate exists.

## Comparison with tested regimes

The closest scientifically coherent tested regime is unit-consistent MLR / IFA gamma0.31. Performance similarity is recorded only as a consistency check and is not used as provenance evidence. No existing or historical-public recipe is an exact evidence-backed match to the published recipe.

## What is directly supported

Paper textual values, all nine public commits, the first-code publication date, current release batch/scheduler/gamma/augmentation/evaluation behavior, and the absence of a public checkpoint-selection rule are directly supported by retained evidence.

## What is inferred

The June release is a cleanup-era snapshot rather than a demonstrated Table 1 snapshot. Unit-consistent gamma0.31 is recommended as the local research baseline because its scheduler semantics are coherent and its IFA direction matches the paper; this is a baseline choice, not a claim about the authors' private recipe.

## What cannot be recovered publicly

The Table 1 internal commit, scheduler step unit, feature augmentation choice, exact 20-trial implementation, and checkpoint-selection/aggregation rule cannot be recovered from public paper-time source because no such source snapshot is public.

## Final exact-recipe assessment

Method-level reproduction is supported; exact published recipe is not publicly recoverable from current release/history.

## Recommended next action

Stop archaeology and send `AUTHOR_QUERY_DRAFT.md` only after user approval. Until new provenance appears, reuse unit-consistent MLR and unit-consistent IFA gamma0.31; do not launch another parameter sweep or infer recipe from performance.
