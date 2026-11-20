"""Test the D435i RGB V4L2 node directly, discarding frame contents."""
from datetime import datetime, timezone, timedelta
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def check(project):
    sys.path.insert(0, str(project / "synchronized_capture"))
    from cameras import supports_rgb
    nodes = []
    for p in Path("/sys/class/video4linux").glob("video*"):
        device = (p / "device").resolve()
        usb = next((u for u in (device,*device.parents) if (u / "product").is_file()),None)
        if usb and "435i" in (usb / "product").read_text().lower() and supports_rgb("/dev/"+p.name):
            nodes.append(("/dev/"+p.name, usb.name))
    if len(nodes) != 1:
        return dict(status="ambiguous_or_missing", count=len(nodes))
    node, port = nodes[0]
    if subprocess.run(["fuser",node],capture_output=True).returncode == 0:
        return dict(status="busy",node=node)
    if not shutil.which("ffmpeg"):
        return dict(status="native_reader_unavailable",node=node)
    formats = subprocess.run(["ffmpeg","-hide_banner","-nostdin","-f","v4l2","-list_formats","all",
                              "-i",node],capture_output=True,text=True,timeout=5).stderr
    fmt=next((f for f in ("yuyv422","mjpeg","uyvy422","rgb24","bgr24")
              if any(f in line.lower() and "640x480" in line for line in formats.splitlines())),None)
    if fmt is None:
        return dict(status="no_confirmed_640x480_format",node=node,port=port)
    result=subprocess.run(["timeout","-k","2s","12s","ffmpeg","-hide_banner","-nostdin",
                           "-loglevel","warning","-progress","pipe:1","-nostats","-f","v4l2",
                           "-input_format",fmt,"-framerate","30","-video_size","640x480","-i",node,
                           "-frames:v","60","-f","null","-"],capture_output=True,text=True,timeout=17)
    frames=[int(v) for v in re.findall(r"(?m)^frame=(\d+)\s*$",result.stdout)]
    return dict(status="finished",node=node,port=port,format=fmt,returncode=result.returncode,
                progress_frames=max(frames) if frames else None,
                diagnostic=[x[:200] for x in result.stderr.splitlines() if re.search("error|timed out|timeout|corrupt|empty|failed",x,re.I)][-5:])


if __name__=="__main__":
    root=Path(sys.argv[1])
    with (root / "synchronized_capture/recorder.lock").open("rb") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result=dict(status="recorder_busy")
        else:
            result=check(root)
    result["time"]=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    print(json.dumps(result))
