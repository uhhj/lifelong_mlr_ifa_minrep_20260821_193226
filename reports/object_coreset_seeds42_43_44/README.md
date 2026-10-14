# OBJECT Random vs Greedy Coreset — seeds 42/43/44

Completed: 2,400 epochs, 633,000 optimizer updates, 24 final epoch99 checkpoints
and 1,200 formal closed-loop episodes. This is the paper-description-aligned
state=True comparison, separate from older state=False OBJECT experiments.

- [Read-only credibility and mechanism review (中文)](execution/20261014_coreset_result_review/REVIEW_REPORT.md)
- [Full results and all task matrices (中文)](execution/20261013_coreset_extension/FINAL_REPORT.md)
- [Original lightweight evidence](execution/20261013_coreset_extension/final_evidence/remote)
- [Review diagnostics](execution/20261014_coreset_result_review/RESULT_REVIEW.json)
- [Measured feature/action coverage](execution/20261014_coreset_result_review/coverage.json)
- [Recorded evaluation RNG comparisons](execution/20261014_coreset_result_review/rng_review.json)
- [Frozen implementation and protocols](../../experiments/object_coreset_seeds42_43_44/README.md)

Mean AUC: Random 76.354167, Coreset 82.916667; paired difference +6.562500
percentage points (sample SD 7.848528). The individual differences are +1.979167,
+15.625000 and +2.083333. Seed43 contributes most of the improvement. Seed44
has worse NBT and worse later task6 performance. This is exploratory evidence,
not a claim of statistical significance or isolated coverage causality.

The review found better geometric coverage but also an IFA-pair difference at
seed42/stage8 and unequal CUDA RNG snapshots in 10 of 30 paired evaluation cells.
All limitations, negative task results and unknown pretraining modality metadata
are retained. Models, feature payloads, demonstrations and credentials are excluded.

## Offline verification

From repository root, using Python (standard library only):

```sh
python reports/object_coreset_seeds42_43_44/execution/20261014_coreset_result_review/review.py
python reports/object_coreset_seeds42_43_44/execution/20261014_coreset_result_review/build_report.py
```

These recompute the metrics from archived episodes, verify original evidence hashes,
and regenerate derived review files. No training, environment, SSH or GPU is invoked.
The coverage/RNG audit scripts instead require the original server inputs and are
included for traceability; their recorded results were computed read-only on CPU.
Original absolute paths in frozen evidence are historical provenance.
