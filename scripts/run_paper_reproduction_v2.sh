#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:?usage: scripts/run_paper_reproduction_v2.sh audit|test|task6|feature-audit|seed42-mlr|mlr-formal|ifa-formal|summarize}"
case "$MODE" in
  audit)
    test -f reports/paper_reproduction_v2/protocol_audit.md
    test -f reports/paper_reproduction_v2/scheduler_semantics.md
    echo "Paper/release/runner audit documents are present."
    ;;
  test)
    "$PYTHON" -m pytest -q tests/test_paper_repro_protocol.py
    for SCHED_ARM in current_legacy batch10_linear_step batch10_constant; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" SMOKE=1 \
        bash scripts/run_object_task6_reconciliation.sh "$SCHED_ARM"
    done
    ;;
  task6)
    echo "Run the four isolated arms, then their evaluation grids; this mode intentionally does not cross the next gate."
    bash scripts/eval_object_task6_zero_shot.sh
    for ARM in current_legacy release18_legacy batch10_linear_step batch10_constant; do
      bash scripts/run_object_task6_reconciliation.sh "$ARM"
      bash scripts/eval_object_task6_reconciliation.sh "$ARM"
    done
    ;;
  feature-audit)
    "$PYTHON" scripts/audit_object_feature_cache.py
    ;;
  seed42-mlr|mlr-formal|ifa-formal)
    echo "$MODE is gate-controlled and is not enabled before the Task6 verdict." >&2
    exit 3
    ;;
  summarize)
    "$PYTHON" scripts/summarize_object_task6_reconciliation.py
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
