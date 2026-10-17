# Five-seed read-only stability and Event analysis

See [REPORT.md](REPORT.md) for findings, [ANALYSIS.json](ANALYSIS.json) for numerical decompositions, and [the preceding summary](../object_five_seed_summary/REPORT.md) for all results.

From the repository root, using Python 3 (standard library only):

```sh
python -X utf8 reports/object_five_seed_readonly_analysis/analyze.py
```

The script checks INPUT_SHA256.json, reads the adjacent published five-seed results and EVENT_DIAGNOSTICS.json, and recomputes paired differences, leave-one-seed-out means, task contributions, admission changes, coverage and IFA-pair comparisons. It writes only ANALYSIS.json. Seed43 repeat1 is not included in the five-seed statistics.

EVENT_DIAGNOSTICS.json is a compact extraction of 60 existing server admission/IFA JSON files, including their source paths and SHA256 values and an input-unchanged check. The complete 60 source files are not bundled here; the published calculation verifies the compact extraction, not a fresh server-side source verification. It contains counts, coverage diagnostics, per-demo counts and selected IFA pairs, not model weights or training payloads. The adjacent summary contains the previously published episode outcome records.

PROVENANCE.json identifies the original local analysis. Only repository-relative paths and input-hash verification were adapted for publication; numerical calculations are unchanged. The report's reproduction command and upstream reference were updated to the repository layout.

No new training, checkpoint loading, environment rollout or scientific intervention was run for this analysis or publication. Coverage changes are mechanism clues, not a causal identification of failure. The five seeds are exploratory; historical hardware differences and shared action RNG remain limitations.
