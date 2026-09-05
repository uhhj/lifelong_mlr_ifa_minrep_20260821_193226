# Own-reference controlled reconciliation

Starting source: a4ed54d8502a10dd41a1a5ec83f9234e97a0071a. Historical runs and reports remain intact.

Question: does correcting the mixed-batch own-language-reference lookup explain the observed IFA deficit?

Only algorithmic change: explicit ifa_own_reference_fix=true uses current-task masked language features, one per current global latent. Default false preserves the historical release objective. Negative reference, pair selection, margin, normalization, coefficient and all optimization choices remain unchanged.

Three corrected IFA seeds42/43/44, full tasks6-9, 100 epochs, batch10, same seed-specific feature caches and released pretrained checkpoint. AdamW1e-4, weight decay0.1, batch-step linear scheduler/warmup500, gamma0.30, lambda0.1, past_data_rate0.5. Fresh separate replay buffers. One GPU, serial execution. No new MLR training: use existing paired formal MLR and release IFA as controls.

Primary evaluation only: ep99, 20 trials, original test.py task/call order and stochastic GMM action. No best-epoch search. Compare matrices and FWT/NBT/AUC using existing metric implementation; report sample standard deviations and paired corrected-minus-release changes. Pair identity must remain unchanged. Task6 has no IFA and its final tensors must match historical IFA exactly; a mismatch pauses the queue for investigation rather than silently accepting a confound.

Prior diagnostic: 24/24 sampled negative-task-first batches had exactly zero IFA gradient with nonzero loss; 6 other-old-first batches had gradient cosine0.34-0.36; 24 correct-first controls matched. These are stratified fixed-checkpoint probes, not causal rollout estimates.

The old report's claim of paper-text-faithfulness is superseded by the reference bug evidence, without altering its recorded results. Paper Eq4 requires the own task reference: https://arxiv.org/html/2603.10929v2#S3.SS2 . A repair is not assumed to recover the paper numbers. Remaining pretraining/release and finite-evaluation uncertainties will be reported honestly.
