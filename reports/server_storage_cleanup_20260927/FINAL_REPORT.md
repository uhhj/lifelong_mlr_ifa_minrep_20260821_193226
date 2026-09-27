# Server storage cleanup

Removed 84 completed Stage A temporary prefix snapshots (20.319 GiB), after verifying 42 PASS completion records and their measurement SHA256 values. Raw measurements were reverified after removal.

Removed Conda downloaded archives and index cache, and two regenerated APT index binary caches. Installed environments and model dependency caches were retained.

## Preserved

Current confirmation assets and all unique historical research checkpoints, failure captures, source/base models, frozen streams, reports and raw measurement results remain. No formal solver, protocol or recovery slot was changed.

## Space and execution state

- `/`: 3.259 GiB free.
- `/data`: 67.913 GiB free.

No research process is currently running after the server restart in no-GPU mode. The previous QUEUE_STATUS RUNNING field is stale, not proof of an active process. No experiment was restarted during cleanup.

Temporary Stage A prefix states now require reconstruction from retained original models and frozen streams if needed; this is recorded rather than claiming all large files remain.

`formal_training_resumed=false`
