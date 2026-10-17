# Five-seed result evidence (LIBERO-OBJECT, +IFA)

Run from repository root with Python 3.9+ (standard library only):

```sh
python -X utf8 reports/object_five_seed_summary/aggregate.py
```

See [REPORT.md](REPORT.md) for the full table and [RESULTS.json](RESULTS.json) for exact fractions, matrices and statistics.
First runs of seeds 40–44 contribute 3,000 episodes. Seed43 repeat1 contributes a separate 600 episodes and is excluded from all five-seed statistics. Standard deviation uses ddof=1.

This directory contains evaluation outcome records, existing summaries/acceptance evidence, and SHA256 provenance only. It contains no model weights, demonstrations, latent features, replay buffers, observation trajectories or new environment executions.

PROVENANCE.json maps byte-identical evidence to its original local archive. SOURCE_SHA256.json uses portable relative paths. The published script differs from the local script only in archive locations and portable output paths; all metric and validation logic is preserved. EC44's acceptance and evidence hashes were read from the server on 2026-10-17.

This is a read-only metric aggregation, not a new acceptance of every training artifact. Existing summaries are cross-checked against 180 episode files; all 3,600 records and input hashes are checked. Original experiment limitations remain: exploratory seeds, hardware differences, and action RNG not isolated per trajectory. No statistical significance claim is made.
