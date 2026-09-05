# Language reference stability

The actual stored feature shape is [1,8,4,512]: batch, time, modality, embedding. This follows CompletePolicy.spatial_encode's concatenation along modality dimension2 and BasePolicy.compute_loss's direct save. Language occupies modality0. Some inherited comments reversed the time/modality sizes; the real indexing is documented here.

All17,763 files in the three frozen feature caches (5,921 per seed) were read. For each task, every language vector at every stored time position was bitwise identical to that task's file0 first language vector after float conversion. Maximum absolute difference:0 for all12 seed/task combinations. The task references were also identical across seeds42/43/44.

Consequently, corrected cur_feat[batch_mask,0,0,:] is equivalent to the fixed own-task language reference for these caches. It does not introduce per-example/time/seed-varying references. The first-time/first-modality indices are both zero, so the historical comment mismatch did not affect the controlled loss indexing or preceding fixed-checkpoint gradient probes.

Evidence: reference_stability.json; executable audit: scripts/audit_reference_stability.py. No cache, model or formal result was modified by this check.
