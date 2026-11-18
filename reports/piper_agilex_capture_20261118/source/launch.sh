#!/usr/bin/env bash
set -e
project=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
source /opt/ros/humble/setup.bash
source "$project/agilex_data_tools/vendor/install/local_setup.bash"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=82
export PYTHONPATH="$project/synchronized_capture${PYTHONPATH:+:$PYTHONPATH}"
flock -n -E 73 "$project/synchronized_capture/recorder.lock" /usr/bin/python3 -B \
    "$project/agilex_data_tools/capture.py" --project "$project" || {
    status=$?
    if [ "$status" -eq 73 ]; then printf '相机正在被旧录制台占用，请先关闭旧录制台窗口。\n'; fi
    read -r -p '按回车关闭窗口。' || true
    exit "$status"
}
