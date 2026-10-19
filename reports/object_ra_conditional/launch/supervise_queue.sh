#!/usr/bin/env bash
set -u
Q=/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019
test ! -e "$Q/wrapper.started" || exit 92
date -Iseconds > "$Q/wrapper.started"
echo $$ > "$Q/wrapper.pid"
/data/mlr_ifa/env/bin/python -u "$Q/queue.py" > "$Q/queue.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
 /data/mlr_ifa/env/bin/python "$Q/final_queue_audit.py" > "$Q/final_queue_audit.log" 2>&1
 rc=$?
fi
echo "$rc" > "$Q/wrapper.exit"
exit "$rc"
