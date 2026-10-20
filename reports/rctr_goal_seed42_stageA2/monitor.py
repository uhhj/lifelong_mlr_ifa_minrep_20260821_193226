"""Read-only status and light artifact export; no model or environment imports."""
import json,os,subprocess,time
from pathlib import Path
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020")
def get(n):
 p=O/n
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except ValueError:return p.read_text()
rows=[json.loads(s) for s in (O/"episodes.jsonl").read_text().splitlines()] if (O/"episodes.jsonl").exists() else []
r=dict(unix=time.time(),status=get("RUN_STATUS.json"),gate=get("ENGINEERING_GATE.json"),
 completed=len(rows),last_episode=rows[-1] if rows else None,
 runner_exit=get("runner.exit"),supervisor_exit=get("supervisor.exit"),analysis=get("ANALYSIS.json"),
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True),
 disk=subprocess.check_output(["df","-h","/data","/root"],text=True),
 processes=subprocess.check_output(["ps","-eo","pid,ppid,etimes,args"],text=True))
r["processes"]=[s for s in r["processes"].splitlines() if "RCTR_GOAL_seed42_stageA2" in s and "monitor.py" not in s and "ps -eo" not in s]
if len(rows)>=5:
 last=rows[-min(30,len(rows)):]
 r["eta_remaining_seconds_estimate"]=(333-len(rows))*sum(x["elapsed_seconds"] for x in last)/len(last)
print(json.dumps(r,indent=2))
