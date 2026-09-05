# Status of the archived historical reports

The files under reports/final_paper_text_faithful are preserved exactly as generated before the own-reference audit. Their numerical results describe the executed historical release-derived implementation. Their assertions of paper-text-faithfulness and method-level NOT_SUPPORTED are not endorsed as conclusions about the paper mechanism.

The reference audit established that the mixed-batch first example can supply another task's language reference. The gradient audit established cancellation when that example belongs to the selected negative task and altered directions for other old-task examples. Paper Eq4 calls for the current task's own reference. This discrepancy invalidates a direct method-level interpretation of the old report, without invalidating its measured rollout numbers.

See PROTOCOL.md and REFERENCE_GRADIENT_AUDIT.md for the controlled correction. Corrected results must be evaluated independently; no improvement is assumed. The existing MLR and release-IFA measurements remain comparison arms, not replaced or overwritten.

Checkpoint loading audit: all24 historical ep99 checkpoints cover the temporal decoder and policy head with matching keys and shapes. CompletePolicy first strictly loads the complete official pretrained state, then applies the partial lifelong checkpoint. Omitted frozen encoder weights therefore retain pretrained initialization. Initial corrected seed42 task6 weights match the historical IFA checkpoint exactly.

Evaluation retains the historical stochastic GMM policy, stage-wise RNG reset, task order, initial-state files, and20 trials in two batches of10 environments. eval.num_procs=20 is present in the launcher, but the actual implementation fixes env_num=10. This is unchanged across arms. Stochastic evaluation and unrecovered paper-time choices remain interpretation limits; three-seed descriptive comparisons are not a formal significance test.
