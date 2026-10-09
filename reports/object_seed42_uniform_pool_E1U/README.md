# Completed LIBERO-OBJECT E1-U, seed42

**COMPLETED; independent acceptance PASS.** 400 epochs,105500 optimizer steps,
28 prescribed checkpoints and880 formal episodes (200 primary,680 secondary).
The formal pipeline ran2026-10-09 12:55:02–19:45:02 Asia/Shanghai (6h50m), exit0.
No E0/E1 retraining, technical rollout retries or seeds43/44 were performed.

| Primary epoch99 metric | E0 | E1 | E1-U |
|---|---:|---:|---:|
| AUC ↑ |77.708333|77.291667|73.645833|
| FWT ↑ |77.500000|76.250000|76.250000|
| NBT ↓ |−0.277778|−3.333333|7.222222|

E1-U−E0 AUC = −4.0625 points; E1−E1-U = +3.645833333 points. The prespecified
condition E1-U AUC strictly above E0 was not met, so no seed extension was started.
The observed single-seed difference does not establish a stable or causal effect.

| Completed stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
|6|35|—|—|—|
|7|55|95|—|—|
|8|15|100|100|—|
|9|30|100|75|75|

Secondary AUC74.375, FWT78.75, NBT10.555555556; selected epochs70/99/95/90.
All tasks and both matrices are in the [full Chinese report](FINAL_REPORT.txt).

## Evidence and implementation

- [Independent acceptance](evidence/INDEPENDENT_ACCEPTANCE.json),
  [final receipt](FINAL_RECEIPT.json) and [frozen contract](evidence/CONTRACT.json).
- [200 primary episodes](evidence/trials_primary.jsonl),
  [680 secondary episodes](evidence/trials_secondary.jsonl) and attempt ledgers.
- [Uniform audit](evidence/UNIFORM_AUDIT.json), [pool diagnostics](evidence/pool_diagnostics.json),
  [full replay flow](evidence/full_replay_flow.json).
- [Full lightweight evidence archive](final_light_evidence.tar.gz):875 original
  evidence files plus their manifest, including400 selection logs with1052400 actual slot IDs,400 pairing
  reports and full training stdout. SHA256 is in PUBLICATION_MANIFEST.json.
- [Read-only validator fix](validator_fix.diff): the original validator rejected
  55.00000000000001 versus55. Its script, exit1 and failure log are retained;
  v2 uses absolute1e-10 only for percentage floats, preserving exact integer counts.
  v2 exit0 and no residual processes establish completion. The historical
  FINAL_ACCEPTANCE.json and raw evidence/FINAL_REPORT.txt retain their earlier
  awaiting-audit status; INDEPENDENT_ACCEPTANCE.json is the final authority.
- [Frozen source, tests and protocol](../../experiments/object_seed42_uniform_pool_E1U/README.md).

All400epoch comparisons match E1 ordered pools, refreshes and attempted positions;
181refreshes,389task pools, exact1/64 weights and ESS64. Pool-level repeats89.2163%
match the uniform-with-replacement expectation89.2142%. This is distinct from
the full replay stream. No invalid-candidate fallback or IFA veto occurred.

Training57.09min, primary evaluation71.05min, secondary281.43min. Scoring remains
in both variants; no scoring-free efficiency claim is made. E0 usedRTX4090,
E1/E1-U RTX3090. E0 historical wall times/buffer snapshots and E1 retained-slot
IDs are unavailable. E1-U−E0 combines pool restriction and replacement structure;
it is not a pure pool-size causal estimate. See report for all limitations.

Publication was explicitly authorized after completion. Historical no-push
statements describe execution-time restrictions and remain unchanged. This
publication contains no weights, latent tensors, demonstrations or buffer tensors.
