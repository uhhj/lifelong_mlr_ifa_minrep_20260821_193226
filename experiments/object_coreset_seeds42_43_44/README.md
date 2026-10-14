# Frozen OBJECT Random vs Greedy Coreset implementation

Upstream base: `a721807f2f7fa7f9833dcc85c592d78e9accd00b` (yfqi/lifelong_mlr_ifa).
Each seed directory contains the actual isolated source snapshot. Its Python,
YAML and shell bytes match `source_sha256` in the corresponding archived CONTRACT.
Seed42 includes its source patch and adjusted paper-aligned protocol; seeds43/44
include their explicit extension patches. Nothing replaces the repository's existing
GOAL or previous OBJECT implementation.

- [Seed42 protocol](seed42/ADJUSTED_PROTOCOL.md)
- [Frozen specification](seed42/paper_aligned_spec.yaml)
- [Extension protocol](EXTENSION_PROTOCOL.md)
- [Results and read-only review](../../reports/object_coreset_seeds42_43_44/README.md)

The snapshots preserve historical server paths and are not automatic launch commands.
Running a new experiment requires separate resources and authorization. This publication
does not resume or execute any experiment. It includes no weights, demonstrations,
latent payloads or authentication material.

All arms use state=True, batch10, four tasks × 100 complete epochs, BC + mean IFA0.1,
gamma0.3, final epoch99 evaluation, and equal ceil(0.5N) storage quotas. The selection
rule differs: Random versus frozen visual/action Greedy K-Center. The official
pretraining state flag is unknown; scheduler details were explicitly completed.
Seeds43/44 were authorized after seed42 results and are exploratory extensions.
