"""Check both PiPER RGB streams without starting a recorder or storing images."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

import numpy as np


def clock():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def emit(event, **values):
    print(json.dumps(dict(event=event, time=clock(), **values), ensure_ascii=False), flush=True)


def percentiles(values):
    return [round(float(x), 3) for x in np.percentile(values, [50, 95, 100])] if len(values) else []


def pair(a, b):
    i = j = 0
    skews, times = [], []
    while i < len(a) and j < len(b):
        delta = a[i] - b[j]
        if abs(delta) <= 20_000_000:
            skews.append(abs(delta) / 1e6)
            times.append(max(a[i], b[j]))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return skews, times


def main(project, duration, selected="both"):
    sys.path.insert(0, str(project / "synchronized_capture"))
    from cameras import CameraStream, supports_rgb

    config = {}
    ports = {}
    seen = set()
    started = time.time()
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        dev = (node / "device").resolve()
        usb = next((p for p in (dev, *dev.parents) if (p / "product").is_file()), None)
        if usb is None:
            continue
        product = (usb / "product").read_text().strip()
        if "realsense" not in product.lower():
            continue
        role = "overhead" if "435i" in product.lower() else "wrist" if "405" in product else None
        if role is None or selected not in ("both", role):
            continue
        if usb.name not in seen:
            emit("usb", role=role, model=product, port=usb.name, speed_mbps=(usb / "speed").read_text().strip())
            seen.add(usb.name)
        devname = "/dev/" + node.name
        if not supports_rgb(devname):
            continue
        if role in config:
            raise RuntimeError(f"Ambiguous RGB nodes for {role}")
        config[role] = dict(kind="v4l2", device=devname, model=product, hardware_id=role)
        ports[role] = usb.name
    if not config:
        emit("no_rgb_nodes")
        return
    with ExitStack() as cleanup:
        lock_path = project / "synchronized_capture/recorder.lock"
        if lock_path.exists():
            lock = cleanup.enter_context(lock_path.open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                emit("recorder_busy")
                return
        busy = subprocess.run(["fuser", *[c["device"] for c in config.values()]], capture_output=True)
        if busy.returncode == 0:
            emit("camera_busy")
            return
        streams = {}
        for role, cfg in config.items():
            streams[role] = CameraStream(cfg)
            cleanup.callback(streams[role].close)
        wait = time.monotonic() + 8
        while time.monotonic() < wait and not all(s.latest() for s in streams.values()):
            time.sleep(.05)
        begin = time.monotonic_ns()
        end = begin + int(duration * 1e9)
        emit("start", roles=list(config), nodes={r:c["device"] for r,c in config.items()}, seconds=duration)
        stamps = {r:[] for r in streams}
        seq = Counter()
        changes = Counter()
        previous_crc = {}
        skipped = Counter()
        shape_errors = Counter()
        contrast = {r:[] for r in streams}
        progress = begin + 30_000_000_000
        while True:
            now = time.monotonic_ns()
            for role, stream in streams.items():
                for frame in stream.frames():
                    n, t = frame["sequence"], frame["host_monotonic_ns"]
                    if n <= seq[role] or t < begin or t >= end:
                        continue
                    if seq[role]:
                        skipped[role] += n - seq[role] - 1
                    seq[role] = n
                    stamps[role].append(t)
                    im = frame["image"]
                    if im.shape != (480, 640, 3):
                        shape_errors[role] += 1
                    sample = im[::12, ::12]
                    crc = zlib.crc32(sample.tobytes())
                    if role in previous_crc and previous_crc[role] == crc:
                        changes[role] += 1
                    previous_crc[role] = crc
                    contrast[role].append(float(np.std(sample)))
            if now >= end:
                break
            if now >= progress:
                emit("progress", elapsed=round((now-begin)/1e9), frames={k:len(v) for k,v in stamps.items()},
                     age_ms={k:round((now-v[-1])/1e6,1) if v else None for k,v in stamps.items()})
                progress += 30_000_000_000
            time.sleep(.025)
        for role, ts in stamps.items():
            gaps = np.diff(ts) / 1e6
            emit("stream", role=role, frames=len(ts), rate_hz=round(len(ts)/duration,3),
                 gaps_ms=percentiles(gaps), gaps_over_50ms=int(np.sum(gaps>50)),
                 gaps_over_100ms=int(np.sum(gaps>100)), sequence_skips=skipped[role],
                 same_sample_neighbors=changes[role], bad_shape=shape_errors[role],
                 image_std=percentiles(contrast[role]), error=streams[role].error,
                 first_last_ms=[round((ts[0]-begin)/1e6,1),round((end-ts[-1])/1e6,1)] if ts else [])
        if len(stamps) == 2:
            skews, pairs = pair(stamps["overhead"], stamps["wrist"])
            emit("pairs", count=len(pairs), skews_ms=percentiles(skews), pair_gap_ms=percentiles(np.diff(pairs)/1e6),
                 first_last_ms=[round((pairs[0]-begin)/1e6,1),round((end-pairs[-1])/1e6,1)] if pairs else [])
    closed = time.time()
    time.sleep(1)
    proc = subprocess.run(["journalctl","-k","--since",f"@{int(started)}","--until",f"@{int(time.time())+1}",
                           "--no-pager","-o","short-unix"], capture_output=True,text=True)
    events = []
    branches = {p for port in ports.values() for p in (port,port.rsplit(".",1)[0])}
    for line in proc.stdout.splitlines():
        if not any(re.search(r"(?<![\d.])"+re.escape(p)+r"(?=[:. ])",line) for p in branches):
            continue
        kind = next((k for k,s in [("disconnect","USB disconnect"),("reset","reset SuperSpeed"),
                     ("unresponsive","not responding"),("uvc_error","Non-zero status"),
                     ("descriptor_error","descriptor read"),("reconnect","new SuperSpeed")] if s in line),None)
        if kind:
            when=float(line.split()[0])
            events.append([round(when-started,3),kind, "shutdown" if when>=closed else "startup" if when<started+3 else "stream_or_close"])
    emit("kernel", journal_rc=proc.returncode, events=events[:50], counts=dict(Counter(e[1] for e in events)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv)>3 else "both")
