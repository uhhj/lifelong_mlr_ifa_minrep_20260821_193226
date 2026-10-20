#!/usr/bin/env bash
set -u
Q=/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020
/data/mlr_ifa/env/bin/python -u "$Q/acquire_libero50_server.py"
rc=$?
printf '%s\n' "$rc" > "$Q/libero50_acquisition.exit"
exit "$rc"
