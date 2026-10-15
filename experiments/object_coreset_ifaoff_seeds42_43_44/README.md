# Frozen IFA-off original-protocol experiment

This additive snapshot is the actual seed42/43/44 implementation used for the completed LIBERO-OBJECT Random/Coreset IFA-off comparison. It does not modify earlier experiments or the top-level model code.

- [Protocol](PROTOCOL.md)
- [Results and offline verification](../../reports/object_coreset_ifaoff_seeds42_43_44/README.md)
- Each seed source matches every source_sha256 entry in its archived CONTRACT.json.
- Each source_vs_ifa_on.patch compares against the corresponding published +IFA source at commit46fb9affd392f95aa7ce097094d0795a2910c4cc.
- shared/ contains the actual preparation and isolated GPU/training acceptance helpers; orchestration/ contains the executed original-budget queue/evaluation/audit scripts.

Only the training loss changes from BC+0.1IFA to BC; IFA diagnostic forward remains. state=True,batch10,100 complete epochs/task,gamma0.3,warmup500 and step-linear schedule,original shuffle/storage quotas,Random/Coreset selectors and original global-RNG evaluator are retained. Four final models per arm each evaluate seen tasks on initial states0–19,200episodes/arm. This does not implement the cancelled9000-rollout plan.

Historical server paths and one-shot protections are intentionally preserved. These are provenance snapshots, not instructions to restart completed runs. Original local audit scripts require the local lightweight tar download; the repository's verify_publication.py works directly on published text evidence. No weights or data payloads are included.
