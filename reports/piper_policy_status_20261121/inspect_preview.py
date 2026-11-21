"""Summarize a running preview from its own log and open FDs, without opening video devices."""
from collections import Counter
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

base = Path(sys.argv[1])
bj = datetime.timezone(datetime.timedelta(hours=8))
def iso(t):
    return datetime.datetime.fromtimestamp(t,bj).isoformat(timespec="milliseconds")

def camera_processes():
    procs=[]
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args=p.read_bytes().split(b"\0")
            idx=next((i for i,a in enumerate(args) if a.endswith(b"/runner.py")),None)
            if idx is None or args[idx+1] not in (b"infer",b"guard"):continue
            devices={}
            for f in (p.parent/"fd").iterdir():
                try:node=os.readlink(f)
                except OSError:continue
                if not node.startswith("/dev/video"):continue
                dev=(Path("/sys/class/video4linux")/Path(node).name/"device").resolve()
                product=next((d/"product" for d in (dev,*dev.parents) if (d/"product").is_file()),None)
                devices[node]=product.read_text().strip() if product else "unknown"
            procs.append(dict(role=args[idx+1].decode(),task_index=args[idx+3].decode(),devices=devices))
        except (OSError,IndexError):continue
    return sorted(procs,key=lambda x:x["role"])

def latest():
    return max((base/"policy_runs").glob("*_B1.jsonl"),key=lambda f:f.stat().st_mtime,default=None)

def stats(x):
    if not x:return None
    z=sorted(x)
    return dict(min=round(z[0],3),median=round(statistics.median(z),3),
                p95=round(z[math.ceil(.95*len(z))-1],3),max=round(z[-1],3))

def preview():
    processes=camera_processes()
    start_wall=time.time()
    a=time.monotonic_ns()
    time.sleep(12)
    b=time.monotonic_ns()
    end_wall=time.time()
    path=latest()
    out=dict(start=iso(start_wall),end=iso(end_wall),duration_s=round((b-a)/1e9,3),
             processes_before=processes,processes_after=camera_processes())
    if path is None:return out
    rows=[]
    events=Counter()
    last=None
    last_pred=None
    with path.open("rb") as f:
        end=os.fstat(f.fileno()).st_size
        while f.tell()<end:
            line=f.readline()
            if not line.endswith(b"\n"):break
            r=json.loads(line)
            events[(r["event"]+"/"+r.get("mode","")).rstrip("/")]+=1
            if r["event"]=="preview":out["session_start"]=iso(r["wall_ns"]/1e9)
            if r["event"]=="prediction" and r["mono_ns"]<=b:last_pred=r
            if r["event"]=="prediction" and a<=r["mono_ns"]<=b:rows.append(r)
            last=r
    out.update(file=path.name,events=events,last_event=last["event"] if last else None,
               end_reason=last.get("detail") if last else None,new=len(rows),
               last_pred_age_ms=round((b-last_pred["mono_ns"])/1e6,3) if last_pred else None)
    if not rows:return out
    out.update(hz=round(len(rows)*1e9/(b-a),3),all_preview=all(r["mode"]=="preview" and "sent" not in r for r in rows),
               all_full=all(r["full"] for r in rows),finite7=all(len(r[k])==7 and all(map(math.isfinite,r[k])) for r in rows for k in ("q","p")),
               latency_ms=stats([(r["mono_ns"]-r["stamp"])/1e6 for r in rows]),
               gap_ms=stats([(y["stamp"]-x["stamp"])/1e6 for x,y in zip(rows,rows[1:])]),
               unique_stamps=len({r["stamp"] for r in rows}), largest_joint_delta_deg=stats([max(abs(math.degrees(x-y)) for x,y in zip(r["p"][:6],r["q"][:6])) for r in rows]),
               gripper_prediction_mm=stats([r["p"][6] for r in rows]),gripper_feedback_mm=stats([r["q"][6] for r in rows]))
    return out

if __name__=="__main__":print(json.dumps(preview(),ensure_ascii=False))
