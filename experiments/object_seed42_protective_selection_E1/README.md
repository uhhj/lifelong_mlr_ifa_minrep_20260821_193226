# OBJECT E1 protective same-task replay selection

This is the exact source/configuration snapshot for the completed seed42
experiment, separate from the GOAL implementation at the repository root.
It preserves the E0 BC+IFA update, optimizer, scheduler, concatenated shuffled
loader and random buffer storage. Only old slots can be reassigned to samples
from the same old task. Current samples and task positions remain unchanged.

Fixed selection settings: probability 0.5 per old slot, head-only gradients,
16 historical references per old task, 64 disjoint candidates per old task,
16 current references, refresh before the first update and every 500 updates,
conflict coefficient 1 and temperature 1. The original IFA language reference
is protected. The selector has independent RNG and resumable state.

## Contents

- `USER_PROTOCOL.txt`: the supplied experiment protocol.
- `source/`: frozen deployed source and required configuration/provenance JSON.
  `source/.hydra/` is the original export's generated metadata snapshot, not an
  extra source configuration. Its later regeneration is explicitly audited.
- `source.patch`: new selection and runner modules. It is an additions patch
  against the archived E0-derived tree, **not** a patch to apply over the root
  GOAL tree. The complete frozen snapshot is authoritative.
- `tests/` and `baseline_evidence/policy_head.py`: original software tests and fixture.
- `recovery/`: the separately disclosed spawn import repair and read-only
  finalizer repair. Neither changed training, action sampling or rollout rules.

The historical E0 report was recovered from commit
`1c64d9051124307e8debfc2c47aced189e32d393`; archived reports are linked from the
[result directory](../../reports/object_seed42_protective_selection_E1/README.md).
Frozen source SHA256 values, input configuration and checkpoint provenance are
in its `evidence/CONTRACT.json` and `PUBLICATION_MANIFEST.json`. This publication
commit is not claimed to be the earlier training source Git commit.

## Actual execution

The run used Python 3.8, PyTorch 1.13.0+cu116 and one RTX3090. The launcher files
retain their real `/data/mlr_ifa/...` paths for traceability. They require the
original server environment, asset symlinks, demonstrations, pretrained weights
and recovered feature cache; these payloads are intentionally not published.
This is an execution archive, not a path-independent one-command installer.

```bash
# Original server commands, not commands to rerun a completed experiment:
bash /data/mlr_ifa/experiment_workspaces/E1_protective_object_20261008/source/protective_e1/export.sh
bash /data/mlr_ifa/experiment_workspaces/E1_protective_object_20261008/source/protective_e1/pipeline.sh
# After a spawn import failure before any environment steps:
bash /data/mlr_ifa/experiments/E1_protective_object_seed42_20261008/engineering_recovery/20261008_import_binding/resume_eval.sh
```

Twenty synthetic software tests and the real GPU preflight passed. The preflight
performed five isolated optimizer calls; those states were discarded before
formal training. The subsequent import-only recovery check performed no rollout.

```bash
# Software-only check, in an environment with compatible torch and numpy:
python tests/test_replay_selection.py
```

Do not invoke the old pipeline against its completed output directory. Training
and environment execution are complete; evaluation recovery retained the original
failure evidence and all successful/failed formal episodes. The final recovery
exit is 0; the original pipeline exit 1 remains as historical evidence.
