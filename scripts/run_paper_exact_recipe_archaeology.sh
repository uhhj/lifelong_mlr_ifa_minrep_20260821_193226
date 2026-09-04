#!/usr/bin/env bash
set -euo pipefail

export PAPER_ARCHAEOLOGY_NO_TRAIN=1

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODE="${1:?usage: $0 clone|inventory|extract|history|paper|compare|summarize|test|formal}"
OUT="$PROJECT_ROOT/reports/paper_exact_recipe_source_archaeology"
LINK="$PROJECT_ROOT/external/source_archaeology/upstream_lifelong_mlr_ifa"
NATIVE="${ARCHAEOLOGY_NATIVE_CLONE:-/dev/shm/upstream_lifelong_mlr_ifa_20260904}"

clone_upstream() {
  mkdir -p "$PROJECT_ROOT/external/source_archaeology"
  if git -C "$LINK" rev-parse HEAD >/dev/null 2>&1; then
    return
  fi
  if [[ ! -d "$NATIVE/.git" ]]; then
    git clone https://github.com/yfqi/lifelong_mlr_ifa.git "$NATIVE"
  fi
  if [[ ! -e "$LINK" ]]; then
    ln -s "$NATIVE" "$LINK"
  fi
  git -C "$LINK" fetch --all --tags --prune
}

inventory() {
  "$PYTHON" scripts/archaeology_inventory_upstream.py --repo "$LINK" --out "$OUT"
}

extract() {
  "$PYTHON" scripts/extract_upstream_recipe_history.py --repo "$LINK" --root "$PROJECT_ROOT" --out "$OUT"
}

case "$MODE" in
  clone) clone_upstream ;;
  inventory) clone_upstream; inventory ;;
  extract|history|paper|compare|summarize) clone_upstream; inventory; extract ;;
  test) "$PYTHON" -m pytest -q tests/test_paper_recipe_archaeology.py ;;
  formal)
    clone_upstream
    inventory
    extract
    "$PYTHON" -m pytest -q tests/test_paper_recipe_archaeology.py
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
