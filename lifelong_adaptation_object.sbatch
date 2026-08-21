#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:-mlr_ifa}"
case "$METHOD" in
  mlr) exec "$PROJECT_ROOT/scripts/run_object_mlr.sh" ;;
  mlr_ifa) exec "$PROJECT_ROOT/scripts/run_object_mlr_ifa.sh" ;;
  *) echo "usage: bash lifelong_adaptation_object.sbatch [mlr|mlr_ifa]" >&2; exit 2 ;;
esac
