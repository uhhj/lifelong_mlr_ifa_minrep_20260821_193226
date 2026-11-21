"""Inspect USB enumeration, video users and camera-related kernel messages."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

project=Path(sys.argv[1])
tz=timezone(timedelta(hours=8))
def clock(t):
    return datetime.fromtimestamp(t,tz).isoformat(timespec="seconds")
def cmd(args):
    return subprocess.run(args,capture_output=True,text=True,timeout=10)
def prop(p,key):
    try:return (p/key).read_text().strip()
    except OSError:return None
def usbdev(p):
    return dict(port=p.name,product=prop(p,"product"),speed=prop(p,"speed"),bus=prop(p,"busnum"),num=prop(p,"devnum"))
usb=[]
for p in Path("/sys/bus/usb/devices").iterdir():
    name=prop(p,"product") or ""
    if "realsense" in name.lower():
        d=usbdev(p)
        d["parents"]=[usbdev(x) for x in p.resolve().parents if (x/"idVendor").exists()]
        usb.append(d)
nodes=[]
for p in sorted(Path("/sys/class/video4linux").glob("video*")):
    dev=(p/"device").resolve()
    parent=next((x for x in (dev,*dev.parents) if (x/"product").is_file()),None)
    if parent and "realsense" in (prop(parent,"product") or "").lower():
        nodes.append(dict(node="/dev/"+p.name,port=parent.name,index=prop(p,"index"),name=prop(p,"name")))
users={}
for p in Path("/proc").glob("[0-9]*/fd"):
    try:
        used=set()
        for f in p.iterdir():
            try:where=os.readlink(f)
            except OSError:continue
            if where.startswith("/dev/video"):used.add(where)
        if used:
            args=(p.parent/"cmdline").read_bytes().decode(errors="replace").split("\0")
            users[p.parent.name]=dict(program=Path(args[0]).name, scripts=[Path(x).name for x in args if x.endswith(".py")],nodes=sorted(used))
    except OSError:continue
logs=[]
for p in sorted((project/"agilex_data_tools/policy_runs").glob("*_B1.jsonl"),key=lambda x:x.stat().st_mtime)[-5:]:
    with p.open("rb") as f:
        size=os.fstat(f.fileno()).st_size
        f.seek(max(0,size-12000))
        ev={}
        for line in reversed(f.read().splitlines()):
            if line.startswith(b"{"):
                try:
                    ev=json.loads(line)
                    break
                except ValueError:
                    continue
    logs.append(dict(file=p.name,time=clock(p.stat().st_mtime),event=ev.get("event"),mode=ev.get("mode"),detail=ev.get("detail")))
j=cmd(["journalctl","-k","--since","30 minutes ago","--no-pager","-o","short-unix"])
lines=[re.sub(r"SerialNumber: \S+","SerialNumber: [omitted]",line[:300]) for line in j.stdout.splitlines()
       if re.search(r"usb |uvcvideo|xhci|video4linux|videobuf|dwc3|bandwidth",line,re.I)]
print(json.dumps(dict(at=datetime.now(tz).isoformat(timespec="seconds"),usb=usb,nodes=nodes,users=users,
                      policy=logs,journal_rc=j.returncode,kernel_lines=len(lines),kernel_tail=lines[-100:]),ensure_ascii=False))
