"""Read the passive CAN archive, B1 starting feedback and the running UI's source."""
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time


def inspect_ui():
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmd.read_bytes().split(b"\0")
            arg = next((os.fsdecode(v) for v in args if Path(os.fsdecode(v)).name == "piper_ui.py"), None)
            if not arg:
                continue
            source = (cmd.parent / "cwd" / arg).read_bytes()
        except OSError:
            continue
        code = source.decode()
        methods = {f.name:ast.get_source_segment(code,f) for f in ast.walk(ast.parse(code))
                   if isinstance(f,ast.FunctionDef) and f.name in ("gripper_ctrl","update_gripper","run_enable")}
        bindings = [line.strip() for line in code.splitlines() if any(token in line for token in (
            "gripper_slider.valueChanged", "gripper_slider = ", "gripper_slider_label = ",
            "gripper_slider.setEnabled", "gripper_slider.setRange", "gripper_combobox =", "if all(x == 1"))]
        return dict(sha256=hashlib.sha256(source).hexdigest(), methods=methods, bindings=bindings)


def expert_starts(work):
    base = work / "expert_sets/B1_top50_20261120_163349"
    manifest = json.loads((base/"manifest.json").read_bytes())
    assert manifest["task"] == "B1" and len(manifest["episodes"]) == 50
    for episode in manifest["episodes"]:
        with (base/episode["episode_dir"]/"alignment.jsonl").open() as stream:
            row = next(json.loads(line) for i,line in enumerate(stream) if i == episode["first_action_row"])
        fb = row["feedback"]
        yield [math.degrees(v) for v in fb["joints_rad"]] + [fb["gripper_mm"]]


def read_feedback(work):
    parts = list((work.parent/"diagnostics").glob("piper_passive_*/frames-*.bin"))
    latest = max(parts, key=lambda p:p.stat().st_mtime_ns)
    files = sorted(latest.parent.glob("frames-*.bin"))[-2:]
    record = struct.Struct("<QQIBBH8s")
    samples = []
    for file in files:
        with file.open("rb") as stream:
            count = stream.seek(0,2)//record.size
            first = max(0,count-14000)
            stream.seek(first*record.size)
            samples.extend(record.iter_unpack(stream.read((count-first)*record.size)))
    newest=max(s[1] for s in samples)
    window=sorted((s for s in samples if 0 <= newest-s[1] <= 2_000_000_000), key=lambda s:s[1])
    counts=Counter()
    data={}
    spans={}
    errors=0
    commands=Counter()
    for wall, mono, cid, port, size, flags, payload in window:
        if port:
            continue
        if cid & 0x20000000:
            errors+=1
        if cid in (0x150,0x151,0x155,0x156,0x157,0x159,0x471):
            commands[hex(cid)]+=1
        if size!=8:
            continue
        counts[cid]+=1
        data[cid]=(wall,mono,payload)
        if cid in (0x2a5,0x2a6,0x2a7,0x2a8):
            axes=struct.unpack(">ii",payload) if cid!=0x2a8 else (struct.unpack_from(">i",payload)[0],)
            for i,v in enumerate(axes):
                k=(cid,i)
                lo,hi=spans.get(k,(v,v))
                spans[k]=(min(lo,v),max(hi,v))
    used=[0x2a5,0x2a6,0x2a7,0x2a8]
    if any(cid not in data for cid in used):
        raise RuntimeError("Incomplete feedback in the latest passive window")
    age=(time.monotonic_ns()-min(data[cid][1] for cid in used))/1e9
    if not 0<=age<3:
        raise RuntimeError(f"Stale feedback ({age:.1f} s old)")
    pose=[v/1000 for cid in used[:3] for v in struct.unpack(">ii",data[cid][2])]
    pose.append(struct.unpack_from(">i",data[0x2a8][2])[0]/1000)
    stamp=max(data[c][0] for c in used)
    clock=datetime.fromtimestamp(stamp/1e9,timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")
    return dict(feedback_at=clock,age_s=round(age,3),pose_deg_mm=pose,
                ranges_2s=[[lo/1000,hi/1000] for _,(lo,hi) in sorted(spans.items())],
                latest_status=data[0x2a1][2].hex() if 0x2a1 in data else None,
                drive_status=[data[i][2][5] if i in data else None for i in range(0x261,0x267)],
                gripper_status=data[0x2a8][2][6], error_frames=errors, commands=commands,
                feedback_count=[counts[i] for i in used])


work=Path(sys.argv[1])
starts=list(expert_starts(work))
ui=inspect_ui()
observed=read_feedback(work)
bounds=[(min(axis),max(axis)) for axis in zip(*starts)]
pose=observed["pose_deg_mm"]
outside=[name for name,v,(lo,hi) in zip(("J1","J2","J3","J4","J5","J6","grip"),pose,bounds) if not lo<=v<=hi]
closest=min(starts,key=lambda s:sum(((v-ref)/(hi-lo))**2 for v,ref,(lo,hi) in zip(pose[:6],s,bounds)))
print(json.dumps(dict(observed=observed, outside=outside, expert_limits=[[round(a,3),round(b,3)] for a,b in bounds],
                      nearest_six_axis_start=[round(v,3) for v in closest], ui=ui),ensure_ascii=False))
