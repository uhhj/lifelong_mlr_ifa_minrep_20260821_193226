"""Rank frozen B1 audits and write a training manifest and candidate ledger."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path


def eligibility(item, audit, motion):
    n, target, feedback = audit["rows"],audit["target"],audit["feedback"]
    can, passive = audit["independent"],audit["passive"]
    reasons = []
    if (n != item["pairs"] or n != feedback["valid"] or n != item["feedback"] or n != audit["unique"]
            or motion["rows"] != n or audit["gap_ms"][0] <= 0 or item["source"] != "direct_can_rx"
            or audit["source"] != item["source"] or item["targets"] != target["valid"]
            or target["valid"]+target["leading"] != n or motion["valid_start"] != target["leading"]
            or target["missing_after"] or min(target["age_ms"][0],feedback["age_ms"][0]) < 0):
        reasons.append("incomplete_or_misaligned")
    if any(s["bad"] or s["different_from_overhead"] or s["unmatched_rows"] or
           s.get("max_filename_error_ns",float("inf")) > 500 or s.get("repeated_pixels") or
           s.get("backwards_or_duplicate") for s in audit["streams"].values()):
        reasons.append("stream_integrity")
    if (can.get("alignment_sha256") != item["hash"] or can.get("target_exact") != target["valid"]
            or can.get("feedback_exact") != n or not can.get("anchors")):
        reasons.append("independent_can")
    holds = passive.get("hold",[])
    if (not passive.get("status_frames") or passive.get("errors") or passive.get("status_faults")
            or len(holds) != 6 or any(not h or not h["tested"] or h["flags"] for h in holds)):
        reasons.append("can_fault_or_hold_screen")
    if not motion["blue_end_found"] or motion["red_end_found"]:
        reasons.append("terminal_location_unconfirmed")
    if motion["release_longest_frames"] < 5:
        reasons.append("clear_release_unconfirmed")
    return reasons


def percentiles(items, key):
    values = sorted({r[key] for r in items})
    counts = Counter(r[key] for r in items)
    result, lower = {}, 0
    for v in values:
        result[v] = (lower+(counts[v]-1)/2)/max(1,len(items)-1)
        lower += counts[v]
    return result


def choose(inventory, measurements):
    candidates = inventory["candidates"]
    if len({i["id"] for i in candidates}) != len(candidates):
        raise ValueError("Duplicate episode ID")
    if set(i["id"] for i in candidates)-measurements.keys():
        raise ValueError("Missing candidate audit")
    ledger, unique_hashes = [], set()
    for item in sorted(candidates,key=lambda e:e["id"]):
        name = item["id"]
        obs = measurements[name]
        if "error" in obs or obs["hash"] != item["hash"]:
            raise ValueError("Incomplete or stale assessment: "+name)
        metrics, audit = obs["metrics"],obs["quality"]
        reasons = eligibility(item,audit,metrics)
        if item["hash"] in unique_hashes:
            reasons.append("duplicate_alignment")
        if not reasons:
            unique_hashes.add(item["hash"])
        row = {k:v for k,v in metrics.items() if k not in ("latest_payload_mtime_ns","release_first_end","rows","red_end_found","blue_end_found","valid_start")}
        row.update(episode_id=name, alignment_sha256=item["hash"], recorded_pairs=item["pairs"],
                   first_action_row=metrics["valid_start"], eligibility_issues=reasons)
        long_tail = metrics["settled_tail_s"] >= 2 and metrics["settled_tail_fraction"] >= .20
        limited_closure = metrics["pre_release_closing_mm"] is None or metrics["pre_release_closing_mm"]<5
        row["motion_flags"] = sum((metrics["clear_grip_cycles"]>1, metrics["mid_pause_excess_s"]>=.5, long_tail, limited_closure))
        row["brief_release_evidence"] = metrics["release_longest_frames"]<10 or metrics["blue_last30"]<15
        ledger.append(row)
    eligible = [r for r in ledger if not r["eligibility_issues"]]
    if len(eligible)<50:
        raise ValueError(f"Only {len(eligible)} complete qualified episodes")
    keys = ["max_gap_ms","p95_skew_ms","p95_joint_tracking_deg"]
    percent = {key:percentiles(eligible,key) for key in keys}
    for r in eligible:
        r["timing_tracking_percentile_mean"] = round(sum(percent[k][r[k]] for k in keys)/3,8)
    eligible.sort(key=lambda r:(r["motion_flags"],r["brief_release_evidence"],sum(r["skips_in_valid"]),
                                r["timing_tracking_percentile_mean"],r["max_gap_ms"],r["episode_id"]))
    episodes = []
    for rank,r in enumerate(eligible,1):
        r["rank"]=rank
        if rank <= 50:
            name=r["episode_id"]
            episodes.append(dict(rank=rank,episode_id=name,episode_dir="episodes/"+name,
                                 alignment_sha256=r["alignment_sha256"], recorded_pairs=r["recorded_pairs"],
                                 first_action_row=r["first_action_row"],last_action_row=r["recorded_pairs"]-1,
                                 action_count=r["recorded_pairs"]-r["first_action_row"]))
    label="B1_top50_"+datetime.fromisoformat(inventory["asof"]).strftime("%Y%m%d_%H%M%S")
    manifest=dict(task="B1",instruction="夹起木块，放进左侧蓝碟",set_name=label,inventory_at=inventory["asof"],
                  indexing="zero_based_inclusive_alignment_jsonl",episode_count=50,
                  total_action_count=sum(r["action_count"] for r in episodes),episodes=episodes)
    return manifest, dict(inventory_at=inventory["asof"],outcome_counts=inventory["groups"],
                          eligible_count=len(eligible),candidates=ledger)


def write(path, doc):
    path.write_bytes((json.dumps(doc, ensure_ascii=False,indent=2)+"\n").encode("utf8"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory",type=Path)
    parser.add_argument("measurements",type=Path)
    parser.add_argument("output_directory",type=Path)
    parser.add_argument("--fresh-audits",type=Path, help="Override archived audits with a complete new pass on the same inventory")
    args=parser.parse_args()
    inventory=json.loads(args.inventory.read_text(encoding="utf8"))
    measurements=json.loads(args.measurements.read_text(encoding="utf8"))
    if args.fresh_audits:
        audits=json.loads(args.fresh_audits.read_text(encoding="utf8"))
        if set(audits) != {item["id"] for item in inventory["candidates"]}:
            raise ValueError("Fresh audits do not cover the frozen candidate pool")
        for name,audit in audits.items():
            measurements[name]["quality"]=audit
    manifest, ledger=choose(inventory,measurements)
    write(args.output_directory/"B1_expert_50.json",manifest)
    write(args.output_directory/"B1_candidate_ranking.json",ledger)
