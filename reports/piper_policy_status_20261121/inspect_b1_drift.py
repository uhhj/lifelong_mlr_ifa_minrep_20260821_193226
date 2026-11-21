"""Align policy targets with CAN receive timestamps and inspect the STOP feedback window."""
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

work = Path(sys.argv[1])
log = [json.loads(s) for s in (work/"policy_runs"/sys.argv[2]).read_bytes().splitlines()]
watch = work.parent/"diagnostics"/sys.argv[3]
anchor = log[0]["wall_ns"] - log[0]["mono_ns"]
begin = next(r["mono_ns"] for r in log if r["event"]=="start") + anchor
finish = log[-1]["mono_ns"] + anchor
assert log[-1]["event"] == "end"
run = [r for r in log if r["event"]=="prediction" and r["mode"]=="execute"]
times = [r["mono_ns"]+anchor for r in run]
NS = 1_000_000_000
wire = struct.Struct("<QQIBBH8s")
bj = timezone(timedelta(hours=8))


def when(stamp):
    return datetime.fromtimestamp(stamp/NS,bj).isoformat(timespec="milliseconds") if stamp else None


def read_window(lo,hi):
    for path in sorted(watch.glob("frames-*.bin")):
        with path.open("rb") as f:
            size=f.seek(0,2)//wire.size
            def record_at(index):
                f.seek(index*wire.size)
                return wire.unpack(f.read(wire.size))
            if not size or record_at(0)[0]>hi or record_at(size-1)[0]<lo:
                continue
            # Allow a short margin for a batch with closely spaced timestamps.
            left=bisect_left(range(size),lo-NS,key=lambda i:record_at(i)[0])
            right=bisect_left(range(size),hi+NS,key=lambda i:record_at(i)[0])
            f.seek(left*wire.size)
            for frame in wire.iter_unpack(f.read((right-left)*wire.size)):
                if lo<=frame[0]<=hi and frame[3]==0:
                    yield frame


xyz={k:[] for k in "xyz"}
errors=Counter()
error_windows=[Counter() for _ in range(4)]
first_error=last_error=first_post_status=first_estop=last_status=stop_echo=None
post_status=Counter()
last_normal=None
first_errors=[]
last_drive_before_stop={}
first_drive_after={}
last_arm_status_before_stop=None
volts={i:[] for i in range(1,7)}
for t,_,cid,_,n,flags,b in read_window(begin-NS,finish+60*NS):
    if cid & 0x20000000:
        if first_error is None:first_error=t
        last_error=t
        kind=hex(cid & 0x1fffffff)
        errors[kind]+=1
        bucket=0 if t<begin else 1 if t<=finish else 2 if t<=finish+500_000_000 else 3
        error_windows[bucket][kind]+=1
        if len(first_errors)<8:first_errors.append([when(t),kind,b.hex()])
        continue
    if n!=8:continue
    if cid==0x150 and b[0]==1 and t>=begin and stop_echo is None:
        stop_echo=t
    if cid==0x2a1:
        if b[1]==0:last_normal=t
        last_status=(t,b.hex())
        if stop_echo is not None and t>=stop_echo:
            post_status[b.hex()]+=1
            if first_post_status is None:first_post_status=t
            if b[1]==1 and first_estop is None:first_estop=t
        elif stop_echo is None:
            last_arm_status_before_stop=t
    if 0x261<=cid<=0x266:
        axis=cid-0x260
        raw_v=struct.unpack_from(">H",b)[0]
        if stop_echo is None:
            last_drive_before_stop[axis]=[when(t),raw_v,b[5]]
            if begin<=t:volts[axis].append(raw_v)
        elif axis not in first_drive_after:
            first_drive_after[axis]=[when(t),raw_v,b[5]]
    if begin<=t<=finish:
        if cid==0x2a2:
            x,y=struct.unpack(">ii",b)
            xyz["x"].append((t,x/1000)); xyz["y"].append((t,y/1000))
        if cid==0x2a3:xyz["z"].append((t,struct.unpack_from(">i",b)[0]/1000))


def coords(t):
    return {k: v[min(bisect_left(v,(t,)),len(v)-1)][1] for k,v in xyz.items() if v}


points=[]
for sec in (0,.5,1,2,3,5,8,10,(times[-1]-begin)/NS):
    stamp=begin+round(sec*NS)
    i=min(bisect_left(times,stamp),len(times)-1)
    row=run[i]
    points.append(dict(t_s=round((times[i]-begin)/NS,3), xyz_mm=coords(times[i]),
        raw_j1_deg=round(math.degrees(row["p"][0]),3),sent_j1_deg=round(math.degrees(row["sent"][0]),3),
        measured_j1_deg=round(math.degrees(row["measured_at_send"][0]),3),
        raw_grip_mm=round(row["p"][6],3)))
first_cross={}
for deg in (1,5,10,30,60):
    first_cross[str(deg)]={}
    for key in ("p","sent","measured_at_send"):
        i=next((i for i,r in enumerate(run) if math.degrees(r[key][0]-run[0]["q"][0])<=-deg),None)
        first_cross[str(deg)][key]=round((times[i]-begin)/NS,3) if i is not None else None

health=[]
for line in (watch/"health.jsonl").read_bytes().splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if begin-6*NS<=r["wall_ns"]<=finish+60*NS and r["kind"]=="health":
        health.append(dict(at=when(r["wall_ns"]),overflow=r.get("overflow",{}),link=r.get("link",{}).get("can0",{})))
j=subprocess.run(["journalctl","-k","--since",f"@{begin//NS-2}","--until",f"@{finish//NS+61}",
                  "-o","short-unix","--no-pager"],capture_output=True,text=True)
terms=("bus-off","bus off","usb disconnect","reset superspeed","reset high-speed","entered error","gs_usb")
kernel=[dict(at=when(round(float(s.split()[0])*NS)),events=[term for term in terms if term in s.lower()])
        for s in j.stdout.splitlines() if any(term in s.lower() for term in terms)]
residual=[math.degrees(r["p"][0]-r["q"][0]) for r in run]
print(json.dumps(dict(task=log[0].get("task"),start=when(begin),end=when(finish),points=points,
    first_negative_cross_s=first_cross,j1_negative_residual_frames=sum(v<0 for v in residual),
    j1_residual_sum_range=[round(f(residual),3) for f in (sum,min,max)],frames=len(run),
    xyz_first_mm={k:v[0][1] for k,v in xyz.items()},xyz_final_mm={k:v[-1][1] for k,v in xyz.items()},
    xyz_limits_mm={k:[min(z for _,z in v),max(z for _,z in v)] for k,v in xyz.items()},
    first_error=when(first_error),last_error=when(last_error),first_errors=first_errors,
    error_counts=[dict(c) for c in error_windows],stop_echo=when(stop_echo),first_status_after_stop=when(first_post_status),
    first_stop_feedback=when(first_estop),last_normal=when(last_normal),
    last_status=[when(last_status[0]),last_status[1]] if last_status else None,post_status=dict(post_status),
    last_arm_status_before_stop=when(last_arm_status_before_stop),
    last_drive_before_stop=last_drive_before_stop,first_drive_after=first_drive_after,
    voltage_raw_ranges={i:[min(v),max(v)] for i,v in volts.items() if v},
    health=health,kernel_events=kernel,journal_rc=j.returncode),ensure_ascii=False))
