#!/usr/bin/env bash
set -euo pipefail
source /data/mlr_ifa/activate.sh
exec /data/mlr_ifa/env/bin/python /data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020/worker_entrypoint.py /data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020/entries/RCTR_OBJECT_43.json
