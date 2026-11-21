"""Read desktop, policy-log and SocketCAN status without transmitting or opening a camera."""
import collections
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time

work = Path(sys.argv[1])
tz = datetime.timezone(datetime.timedelta(hours=8))
def clock(ts):
    return datetime.datetime.fromtimestamp(ts,tz).isoformat(timespec="milliseconds")

def read_cmd(args, env=None):
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=4)

env = dict(os.environ)
env["XDG_RUNTIME_DIR"] = "/run/user/"+str(os.getuid())
env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path="+env["XDG_RUNTIME_DIR"]+"/bus"
for line in read_cmd(["systemctl","--user","show-environment"],env).stdout.splitlines():
    k,_,v = line.partition("=")
    if k in ("DISPLAY","XAUTHORITY","WAYLAND_DISPLAY"): env[k]=v
keywords = ("piper","armrobot","armcomm","松灵","机械臂","遥操","策略执行","急停","录制","采集")
def relevant(s):
    return any(k in s.lower() for k in keywords)

def windows():
    cmd = ["wmctrl","-l"] if shutil.which("wmctrl") else ["xwininfo","-root","-tree"]
    r=read_cmd(cmd,env)
    lines=r.stdout.splitlines()
    return dict(ok=r.returncode==0, titles=[line.strip() for line in lines if relevant(line)][:16])

def processes():
    kinds=set()
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try: args=p.read_bytes().decode(errors="replace").split("\0")
        except OSError: continue
        for i,a in enumerate(args):
            name=Path(a.replace("\\","/")).name.lower()
            if name in ("desktop.py","stop_panel.py","piper_ui.py","capture.py","candump","piper_watch.py","armrobotua.exe","armrobotua","data_tools_datacapture"):
                kinds.add(name)
            if name=="runner.py" and i+1<len(args) and args[i+1] in ("infer","guard"):
                kinds.add("runner/"+args[i+1])
    return sorted(kinds)

def latest_log():
    files=list((work/"policy_runs").glob("*.jsonl"))
    if not files:return None
    p=max(files,key=lambda f:f.stat().st_mtime)
    counts=collections.Counter()
    detail=None
    last=None
    with p.open() as f:
        for line in f:
            try:r=json.loads(line)
            except ValueError:continue
            last=r.get("event")
            counts[(r.get("event","?")+"/"+r.get("mode","")).rstrip("/")]+=1
            if last=="end":detail=[r.get("outcome"),r.get("detail")]
    return dict(name=p.name,mtime=clock(p.stat().st_mtime),events=counts,last=last,end=detail)

def link():
    r=read_cmd(["/usr/sbin/ip","-j","-d","-s","link","show","dev","can0"])
    if r.returncode:return dict(found=False)
    d,=json.loads(r.stdout)
    info=d["linkinfo"]["info_data"]
    stats=d.get("stats64",d.get("stats",{}))
    return dict(found=True,flags=d["flags"],state=info.get("state"),modes=info.get("ctrlmode"),
                rate=info.get("bittiming",{}).get("bitrate"),rx=stats.get("rx",{}).get("packets"),
                tx=stats.get("tx",{}).get("packets"),ifindex=d["ifindex"])

def sample():
    count=collections.Counter()
    kinds=collections.Counter()
    drives=collections.defaultdict(collections.Counter)
    grips=collections.Counter()
    errors=collections.Counter()
    mode=collections.Counter()
    origin=collections.Counter()
    joint_span={}
    drops=0
    with socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW) as sock:
        sock.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_ERR_FILTER,struct.pack("=I",0x1fffffff))
        overflow=getattr(socket,"SO_RXQ_OVFL",40)
        sock.setsockopt(socket.SOL_SOCKET,overflow,1)
        sock.bind(("can0",))
        sock.settimeout(.1)
        start=time.monotonic()
        while time.monotonic()-start<4:
            try:raw,anc,flags,_=sock.recvmsg(16,128)
            except socket.timeout:continue
            for level,key,b in anc:
                if level==socket.SOL_SOCKET and key==overflow:drops=struct.unpack_from("=I",b)[0]
            cid,n,b=struct.unpack("=IB3x8s",raw)
            if cid & 0x20000000:
                errors[hex(cid)]+=1
                continue
            count[hex(cid)]+=1
            if cid in (*range(0x150,0x160),0x470,0x471):
                origin["local" if flags&socket.MSG_DONTROUTE else "bus"]+=1
            if n!=8:continue
            if cid==0x2a1:kinds[b.hex()]+=1
            if cid in range(0x261,0x267):drives[str(cid-0x260)][hex(b[5])]+=1
            if cid==0x2a8:grips[b[6:].hex()]+=1
            if cid==0x151:mode[b.hex()]+=1
            if cid in (0x2a5,0x2a6,0x2a7):
                for j,v in enumerate(struct.unpack(">ii",b)):
                    axis=1+2*(cid-0x2a5)+j
                    lo,hi=joint_span.get(axis,(v,v))
                    joint_span[axis]=(min(lo,v),max(hi,v))
    return dict(seconds=round(time.monotonic()-start,3),counts=count,status=kinds,drives=drives,
                gripper_status_mode=grips,mode_command=mode,command_origin=origin,errors=errors,
                dropped=drops,joint_spans_deg={k:(hi-lo)/1000 for k,(lo,hi) in joint_span.items()})

r=dict(start=clock(time.time()),windows=windows(),processes=processes(),link_before=link())
try:
    r["binding_file"]= (work/"can0_stop_binding.json").is_file()
    if r["link_before"].get("found") and "UP" in r["link_before"]["flags"]:
        sys.path.insert(0,str(work))
        from stop_link import snapshot, BINDING
        try: r["binding_matches"]=snapshot()[0]==json.loads(BINDING.read_text())
        except Exception as e:r["binding_error"]=str(e)
        r["sample"]=sample()
except OSError as e:r["sample_error"]=str(e)
r.update(link_after=link(),policy_log=latest_log(),end=clock(time.time()))
print(json.dumps(r,ensure_ascii=False))
