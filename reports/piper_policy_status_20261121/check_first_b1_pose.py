"""Compare the latest archived preview with the live release's guard and B1 starts."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

work=Path(sys.argv[1])
base=work/"expert_sets/B1_top50_20261120_163349"
def installed_runner():
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:args=p.read_bytes().split(b"\0")
        except OSError:continue
        for a,b in zip(args,args[1:]):
            if a.endswith(b"/runner.py") and b==b"guard":return Path(a.decode())
    raise RuntimeError("No running policy guard")
runner=installed_runner()
sys.path.insert(0,str(runner.parent))
from bus import LIMITS,bounded_target

path=max((work/"policy_runs").glob("*_B1.jsonl"),key=lambda p:p.stat().st_mtime)
with path.open("rb") as f:
    f.seek(max(0,path.stat().st_size-32000))
    rows=f.read().splitlines()[1:-1]
last=next(json.loads(l) for l in reversed(rows) if json.loads(l).get("event")=="prediction")
q=last["q"]
try:
    bounded_target(last["p"],q,q,.033)
    verdict="pass"
except ValueError as exc:verdict=str(exc)
manifest=json.loads((base/"manifest.json").read_text())
assert manifest["task"]=="B1" and len(manifest["episodes"])==50
starts=[]
outside=0
for ep in manifest["episodes"]:
    with (base/ep["episode_dir"]/"alignment.jsonl").open() as f:
        row=next(json.loads(l) for j,l in enumerate(f) if j==ep["first_action_row"])
    j2,j3=map(math.degrees,row["feedback"]["joints_rad"][1:3])
    starts.append((j2,j3))
    if j2<LIMITS[1][0]-1 or j3>LIMITS[2][1]+1:outside+=1
print(json.dumps(dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
    mode=last["mode"],age_s=round((time.monotonic_ns()-last["mono_ns"])/1e9,3),
    joint_deg=[round(math.degrees(x),3) for x in q[:6]],width_mm=round(q[6],3),
    verdict=verdict,guard_sha256=hashlib.sha256((runner.parent/"bus.py").read_bytes()).hexdigest(),
    starts_j2_j3=[dict(min=round(min(a),3),median=round(statistics.median(a),3),max=round(max(a),3)) for a in zip(*starts)],
    starts_outside_j2_j3=outside,total=len(starts)),ensure_ascii=False))
