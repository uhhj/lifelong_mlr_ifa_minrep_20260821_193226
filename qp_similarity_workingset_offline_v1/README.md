# Similarity-guided working-set CPU feasibility v1

Frozen primary tau=0.999; sensitivity 0.99/0.9999. No model loading, writeback,
GPU requirement, formal solver modification, or training continuation.
The implementation commit is 9c5a05cf68607f19de47dfb778ea2611bf83c928.

The existing kernel and numeric checker are imported without modification from
qp_workingset_refinement_diag_v1. Real asset paths are fixed in that package's
run.FIXTURES and in reports/qp_similarity_workingset_offline_v1/FIXTURES.json.
No saved real x initializes any comparison arm.

## Read results

Start with reports/qp_similarity_workingset_offline_v1/RESULTS_ZH.md,
SUMMARY.json, per_instance.csv, paired_ratios.csv and timing.csv.
raw_evidence.tar.gz contains inputs, evaluation-only references, candidate
arrays, per-repeat JSON, round/linear-system traces, and excluded original
measurements from the fixed timing repair. Extract it in the report directory
before running independent_audit.py. No model checkpoints are included.

## Resume

From the repository and existing environment:

```bash
source .migration/activate.sh
python -m unittest qp_similarity_workingset_offline_v1.tests -v
python -m qp_similarity_workingset_offline_v1.run
```

Completed per-arm repeats and cells are reused, not overwritten. Source hashes
must match PROTOCOL.json. The UNIT_TESTS.json receipt must match tests.py.
A clean rerun uses --output with a new directory containing a fresh unit-test
receipt (passed/tests_run/failures/errors/tests_sha256), instead of deleting
existing results. Pin administration to a different physical CPU from the
single CPU used by the runner. Do not run Git compression or unrelated compute
on the measured CPU. The fixed default runner chooses its first allowed CPU.

The original timing repair is documented in TIMING_ISOLATION_NOTE.json;
it replaced all measurements in the named cells, not whichever timing was faster.
The repair helper is a one-time administrative script, not needed for clean runs.

A negative H3 result must not be replaced with the best sensitivity threshold.
