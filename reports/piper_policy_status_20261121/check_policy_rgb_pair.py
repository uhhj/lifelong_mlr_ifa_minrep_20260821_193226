"""Cold-open the two policy cameras and exercise the installed preview pairing function."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


def block_can(event, args):
    if event == "socket.__new__" and args[1] == socket.AF_CAN:
        raise RuntimeError("Camera-only test")


sys.addaudithook(block_can)
project = Path(sys.argv[1])
sync = project / "synchronized_capture"
desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip())
entry = (desktop / "PiPER-Policy.desktop").read_text()
launch = next(s[5:] for s in entry.splitlines() if s.startswith("Exec="))
release = Path(shlex.split(launch)[-1]).parent
sys.path[:0] = [str(release), str(sync)]
from cameras import CameraStream, configured_cameras
from runner import next_pair

tz = timezone(timedelta(hours=8))
with ExitStack() as cleanup:
    lock = cleanup.enter_context((sync / "recorder.lock").open("rb"))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = configured_cameras()
    roles = ("external", "wrist")
    nodes = [config[r]["device"] for r in roles]
    users = subprocess.run(["fuser", *nodes], capture_output=True)
    if users.returncode != 1:
        raise RuntimeError("Camera busy or ownership check failed")
    streams = []
    started = time.monotonic()
    for role in roles:
        stream = CameraStream(config[role])
        streams.append(stream)
        cleanup.callback(stream.close)
    seq = (0, 0)
    skew = []
    last = started
    first = None
    max_gap = 0
    while time.monotonic() - started < 12:
        pair = next_pair(streams, seq)
        now = time.monotonic()
        if pair:
            if first is None:
                first = now
            else:
                max_gap = max(max_gap, now - last)
            seq = tuple(frame["sequence"] for frame in pair)
            skew.append(abs(pair[0]["host_monotonic_ns"] - pair[1]["host_monotonic_ns"]) / 1e6)
            last = now
        elif now - last > 3:
            break
        time.sleep(.002)
    tail = time.monotonic() - last
    print(json.dumps(dict(at=datetime.now(tz).isoformat(timespec="seconds"),
        nodes=nodes, pairs=len(skew), first_ms=round(1000*(first-started),1) if first else None,
        max_gap_ms=round(1000*max_gap,1), tail_ms=round(1000*tail,1),
        max_skew_ms=round(max(skew),3) if skew else None,
        timed_out=tail>3, camera_errors=[s.error for s in streams])), flush=True)
