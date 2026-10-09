# OBJECT E1-U: uniform sampling within the same candidate pool

Frozen source/configuration for the completed seed42 experiment. It derives from
`e963d8d0f3a4626365d33bdb5c83a60821fff115`'s isolated OBJECT E1 tree, not the
root GOAL implementation. Source hashes are checked against the launch contract.

The only algorithm change is `pool_weighting=uniform`: the same ordered 64-member
candidate pool is sampled uniformly with replacement using the same independent
`random.Random.choices` calls. Scoring remains a side pass. H16/C64/N16, refresh
every500 updates, the 50% same-task replacement gate, current slots, task quotas,
BC+IFA, gamma0.3, OBJECT use_state=False and random storage are retained.
Engineering additions record actual loader IDs, pairing, attempts and costs.

- `source/`: complete source snapshot; historical `.hydra/` export metadata is
  included for provenance but excluded from the frozen active-source hash scope.
- `source.patch`: delta from the archived E1 source, not a root GOAL patch.
- `tests/`: 28 software checks; `baseline_evidence/policy_head.py` is a fixture.
- `USER_PROTOCOL.txt`, `PROTOCOL_AUDIT.txt`, `COMMANDS.txt`: original contract,
  disclosures, real commands and deployment details.
- [Results and independent acceptance](../../reports/object_seed42_uniform_pool_E1U/README.md).

The original server pipeline was `bash protective_e1/pipeline.sh`, using
Python3.8/PyTorch1.13.0+cu116 on RTX3090. It returned0 after400epochs and880episodes.
The28 software checks, exactly5 isolated GPU optimizer calls and import-only spawn
check passed before formal launch. No new tests or rollouts were run to publish.

Files retain real server paths and require external weights, recovered features,
assets, demonstrations and environment dependencies. This is an execution archive,
not a standalone installer. Do not rerun against completed output directories.
Weights, features and tensor buffer payloads are excluded from Git.
