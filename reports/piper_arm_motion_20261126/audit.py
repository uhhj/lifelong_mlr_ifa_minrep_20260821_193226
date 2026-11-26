"""Recompute a redacted incident timeline from private host and desktop observations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def audit(root):
    def read(name):
        return json.loads((root / (name + ".json")).read_bytes())

    inventory = read("baseline")
    files = read("extract")
    snap = read("state")
    gui = read("verify")
    extra = read("locate")
    history = next(n["lines"] for n in gui["log_nodes"] if "Found 1 ports" in n["lines"])
    transitions = []
    for line in history:
        if "暂无周期反馈" in line:
            transitions.append("no_periodic_feedback")
        if "已收到 CAN 报文" in line:
            transitions.append("can_traffic_seen")
        if line.strip() == "[Info]: Arm disable.":
            transitions.append("disable_handler_completed")
    assert transitions == [
        "no_periodic_feedback", "can_traffic_seen", "no_periodic_feedback", "disable_handler_completed"
    ]
    no_feedback = any("无实时反馈" in node["lines"] for node in gui["log_nodes"])
    assert no_feedback
    trace = snap["journal"][2]
    find_time = next(l.split()[0] for l in trace if "COMMAND=/usr/bin/sh" in l and "find_all_can_port.sh" in l)
    query_time = next(l.split()[0] for l in trace if "COMMAND=/usr/sbin/ethtool -i can0" in l)
    ui_rows = [l for l in inventory["relevant_processes"] if "piper_ui.py" in l]
    assert len(ui_rows) == 1
    assert not files["new"] and not files["journal"] == [] and not extra["appjournal"]
    assert not snap["failure"] and snap["counts"] == {} and snap["latest"] == {}
    assert len(snap["errors"]) == 1
    ((error, count),) = snap["errors"].items()
    can_id, payload = error.split()
    assert int(can_id, 16) == 0x24 and bytes.fromhex(payload)[1] == 0x30
    assert count == snap["snapshot_limit"]
    def state(row):
        link = next(item for item in row if item["ifname"] == "can0")
        return link["linkinfo"]["info_data"]["state"]
    return {
        "operator_report": {"arm": "follower/operating arm", "motion": "severe unexpected movement",
                            "reported_window_local_approx": ["15:41", "15:52"],
                            "drives_disabled_at_onset": "unconfirmed", "power_disconnected_afterwards": True},
        "manual_ui_start_local": " ".join(ui_rows[0].split()[1:6]),
        "port_scan_local": find_time, "port_information_read_local": query_time,
        "ui_observed_at": gui["now"], "ui_source_sha256": gui["ui_sha256"],
        "ui_transitions_in_display_order_without_individual_timestamps": transitions,
        "enable_handler_message_in_visible_log": any("Arm enable." in l for l in history),
        "ui_reports_no_fresh_feedback": no_feedback,
        "newest_policy_log_mtime": max(x[0] for x in inventory["policy"] if x[1].endswith(".jsonl")),
        "newest_onsite_stop_log_mtime": files["stops"][0],
        "first_interface_observation": [inventory["now"],state(json.loads(inventory["interfaces"]["text"]))],
        "post_event_receive_only": {"start":snap["start"],"end":snap["end"],
            "interface_state_before_after":[state(snap["before"]),state(snap["after"])],
            "selected_application_frames":sum(snap["counts"].values()),"error_sample_count_capped":count,
            "controller_error_and_no_ack":can_id,"controller_status_byte":hex(bytes.fromhex(payload)[1])},
        "recovery_of_incident_time_can_trace": False,
        "inputs_sha256": {name:sha256((root/(name+".json")).read_bytes()).hexdigest()
                          for name in ("baseline","extract","state","verify","locate")}
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("evidence_directory", type=Path)
    print(json.dumps(audit(cli.parse_args().evidence_directory),ensure_ascii=True,indent=2))
