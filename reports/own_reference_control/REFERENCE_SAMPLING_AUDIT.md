# IFA own-reference sampling audit

Source: server `dm-MS-7D70`, repository `/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`, formal commit `a4ed54d8502a10dd41a1a5ec83f9234e97a0071a`.

Executed CPU-only with PyTorch 1.13.0+cu116:

```bash
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python /tmp/audit_reference_sampling.py
```

No model was loaded, no optimizer step was performed, and no feature, buffer, checkpoint, source, or formal report was modified. The diagnostic script was uploaded to /tmp. Local script: `audit_reference_sampling.py`.

## Reconstruction and validation

Sorted feature paths were sampled with `random.Random(seed)` at ceil(0.5*N) per stage. The capacity branch was verified inactive. Replay filenames were reconstructed with the official task prefix. The stage-specific replay and current paths were shuffled using the experiment seed. DistributedSampler used its actual default seed 0, one replica, rank 0, shuffle enabled and drop_last false; each of 100 epochs used set_epoch(epoch). Batches of size 10 included the final partial batch.

All 1,200 IFA stage/epoch batch counts and optimizer start/end counts matched formal lr_protocol.csv. All 1,200 logged epoch mean losses were finite. All three reconstructed final replay filename sets exactly matched the formal buffers. Per-batch IDs were not originally logged: this is deterministic index reconstruction corroborated by logs and final replay, not a stored execution trace. Feature contents were not examined.

## Results

Eligible means a stage with a logged selected IFA pair and a batch containing current-task samples. Wrong means the first sample's task differs from the current task.

| Seed | Stage | Pair | Wrong / eligible batches | Wrong batch % | Affected current samples / total | Affected sample % | First sample equals negative task |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 7 | (6,7) | 7844 / 22664 | 34.6100 | 46336 / 148000 | 31.3081 | 7844 |
| 42 | 8 | (7,8) | 15212 / 30807 | 49.3784 | 69579 / 156000 | 44.6019 | 7318 |
| 43 | 7 | (6,7) | 7772 / 22668 | 34.2862 | 45800 / 148000 | 30.9459 | 7772 |
| 44 | 7 | (6,7) | 7692 / 22662 | 33.9423 | 45218 / 148000 | 30.5527 | 7692 |

Stages 6 and 9 have no selected pairs for every seed. Stage 8 has no selected pairs for seeds43/44. These stages are ineligible, not zero-mismatch evidence of correct active IFA.

## Interpretation

Reference source task mismatch is frequent in the reconstructed formal batches. The existing code takes current-task global latents after masking but repeats the first unmasked mixed-batch reference. This violates the own-task reference requirement on the counted batches. No numerical equality of reference vectors, gradient cancellation, or causal effect on rollout metrics was established by this filename-only audit.

Next: read-only matched-batch loss/gradient comparison, including direct reference-vector comparison. Preserve original formal results; do not reinterpret them as a faithful implementation-level disproof of paper IFA.
