"""Read-only metadata snapshot. No policy loads, environment calls or training."""
import json
import shutil
import subprocess
import time
from pathlib import Path

Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def run():
    entries=read(Q/"REGISTRY.json") or {}
    entries.setdefault("RCTR_OBJECT_42",dict(
        output="/data/mlr_ifa/experiments/RCTR_OBJECT_seed42_full_20261021"))
    result=dict(unix=time.time(),runs={},queue_status=read(Q/"RUN_STATUS.json"),
                decision=read(Q/"DECISION.json"))
    for key,e in entries.items():
        out=Path(e["output"])
        if not out.exists():
            continue
        epochs=read(out/"ra_coreset/epochs.json") or []
        row=dict(status=read(out/"RUN_STATUS.json"),epochs=len(epochs),
                 optimizer_steps=epochs[-1]["steps"] if epochs else 0,
                 last_epoch=epochs[-1] if epochs else None,
                 exits={p.name:p.read_text().strip() for p in out.glob("*.exit")},
                 formal_records=0,formal_attempt_events=0,collections=[])
        for p in (out/"ra_coreset/evaluation").glob("trials*.jsonl"):
            row["formal_records"]+=sum(bool(line) for line in p.read_text().splitlines())
        for p in (out/"ra_coreset/evaluation").glob("attempts*.jsonl"):
            row["formal_attempt_events"]+=sum(bool(line) for line in p.read_text().splitlines())
        for method in ("rctr","covr"):
            for stage in sorted((out/method).glob("*")):
                if not stage.is_dir():
                    continue
                committed=[read(p) for p in (stage/"committed").glob("*.json")]
                row["collections"].append(dict(stage=stage.name,method=method,
                    branches=len(committed),successful_branches=sum(bool(r["success"]) for r in committed),
                    committed_policy_steps=sum(r["steps"] for r in committed),
                    acceptance=read(stage/"ACCEPTANCE.json")))
        result["runs"][key]=row
    result["disk"]={p:dict(zip(("total","used","free"),shutil.disk_usage(p)))
                    for p in ("/data","/root")}
    result["gpu"]=subprocess.check_output(["nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used","--format=csv,noheader"],text=True).strip()
    result["processes"]=subprocess.check_output(["ps","-eo","pid,ppid,etime,args"],text=True).splitlines()
    result["processes"]=[x for x in result["processes"]
        if any(m in x for m in ("rctr_exp.","caet_exp.","covr_exp.","queue_controller.py"))]
    return result


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
