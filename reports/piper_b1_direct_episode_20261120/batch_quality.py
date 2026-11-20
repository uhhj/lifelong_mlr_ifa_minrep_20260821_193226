"""Audit a saved B1 episode. Run beside check_can.py with WORK and EPISODE arguments."""
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from check_can import RECORD, check


def image_files(folder):
    return {p.stem: p for p in folder.iterdir() if p.stem.replace(".", "").isdigit()}


def stats(values):
    return [round(float(x), 3) for x in np.percentile(values, [0, 50, 95, 100])]


def inspect_streams(episode, rows, keys):
    report = {}
    first_set = None
    for role, path in (("overhead", "camera/color/overhead"), ("wrist", "camera/color/wrist"),
                       ("target", "array/float32/follower_target"), ("feedback", "array/float32/follower_feedback")):
        files = image_files(episode / "episode0" / path)
        indices = set(keys)
        issues = dict(missing=len(indices-files.keys()), extra=len(files.keys()-indices), bad=0)
        if first_set is None:
            first_set = set(files)
        issues["different_from_overhead"] = len(first_set.symmetric_difference(files))
        # Epoch in the file name has six decimals. At a half microsecond,
        # the two neighbouring names can represent the same integer-ns sample.
        actual = sorted(files, key=Decimal)
        paired = keys
        if len(actual) == len(rows):
            residual = [int(Decimal(k)*10**9)-r["sample_monotonic_ns"] for k,r in zip(actual,rows)]
            issues["max_filename_error_ns"] = max(map(abs,residual))
            if issues["max_filename_error_ns"] <= 500:
                paired = actual
        issues["unmatched_rows"] = sum(k not in files for k in paired)
        previous = None
        for key, row in zip(paired, rows):
            if key not in files:
                continue
            if role in ("overhead", "wrist"):
                image = cv2.imread(str(files[key]))
                if image is None or image.shape != (480, 640, 3):
                    issues["bad"] += 1
                elif previous is not None and np.array_equal(image, previous):
                    issues["repeated_pixels"] = issues.get("repeated_pixels", 0) + 1
                previous = image
            else:
                with_value = row[role]
                expected = [] if with_value is None else [*with_value["joints_rad"], with_value["gripper_mm"]]
                array = np.load(files[key], allow_pickle=False)
                try:
                    values = array["data"].ravel() if isinstance(array, np.lib.npyio.NpzFile) or array.dtype.names else array.ravel()
                    if values.shape != (len(expected),) or not np.allclose(values, expected, rtol=1e-7, atol=1e-9):
                        issues["bad"] += 1
                finally:
                    if isinstance(array, np.lib.npyio.NpzFile):
                        array.close()
        if role in ("overhead", "wrist"):
            diff = np.diff([r["cameras"][role]["sequence"] for r in rows])
            issues.update(skipped=int(np.maximum(diff-1, 0).sum()), backwards_or_duplicate=int(np.sum(diff <= 0)))
        report[role] = issues
    return report


def plate_and_block(image, role):
    # Identify the blue dish or red starting paper in this fixed overhead setup.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    y, x = np.indices(h.shape)
    if role == "blue":
        region = (h >= 88) & (h <= 135) & (s > 25) & (v > 52) & (x < 310) & (y > 100) & (y < 335)
    else:
        region = ((h < 12) | (h > 165)) & (s > 70) & (v > 65) & (x > 135) & (x < 450) & (y > 255) & (y < 445)
    mask = cv2.morphologyEx(region.astype("uint8"), cv2.MORPH_CLOSE, np.ones((5, 5), "uint8"))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    hull_area = cv2.contourArea(hull)
    if not ((5500 <= hull_area <= 33000) if role == "blue" else (900 <= hull_area <= 12000)):
        return None
    inner = np.zeros(h.shape, "uint8")
    cv2.drawContours(inner, [hull], -1, 255, -1)
    inner = cv2.erode(inner, np.ones((5, 5), "uint8"))
    wood = (h >= 5) & (h <= 38) & (s >= 22) & (s < 165) & (v > 44) & (v < 228) & (inner > 0)
    wood = cv2.morphologyEx(wood.astype("uint8"), cv2.MORPH_OPEN, np.ones((3, 3), "uint8"))
    parts, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for part in parts:
        area = cv2.contourArea(part)
        (_, _), (w, height), _ = cv2.minAreaRect(part)
        if 240 <= area <= 2300 and min(w, height) >= 14 and max(w, height) <= 66 and min(w,height) >= .47*max(w,height) and area >= .58*w*height:
            candidates.append(round(area))
    return max(candidates, default=0)


def endpoints(episode):
    images = image_files(episode / "episode0/camera/color/overhead")
    keys = sorted(images, key=Decimal)
    out = {}
    for label, indices in (("first", (0, 5, 10, 15, 20, 25)), ("last", (-11, -6, -1))):
        frames = [cv2.imread(str(images[keys[i]])) for i in indices]
        out[label] = {role: [plate_and_block(im, role) for im in frames] for role in ("blue", "red")}
    return out


def camera_log(episode):
    # Scope this corroboration to the recorded launch and its immediate teardown.
    start = datetime.strptime(episode.name[:15], "%Y%m%d_%H%M%S")
    stop = datetime.fromtimestamp((episode/"episode_summary.json").stat().st_mtime)
    if stop < start or stop-start > timedelta(minutes=5):
        return dict(time_window_valid=False)
    cmd = ["journalctl", "-k", "--since", start.isoformat(" "), "--until",
           (stop+timedelta(seconds=5)).isoformat(" "), "--no-pager", "-o", "cat"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    lines = [line for line in proc.stdout.splitlines() if re.search(r"uvcvideo|usb [0-9]|xhci", line, re.I)]
    return dict(journal_exit=proc.returncode, usb_uvc_lines=len(lines),
                disconnect_reset=sum(bool(re.search(r"disconnect|reset .*usb device|not responding", line, re.I)) for line in lines),
                uvc_status=sum("non-zero status" in line.lower() for line in lines))


def passive_window(segments, start, end):
    chunks = []
    matching = []
    for p in segments:
        n = p.stat().st_size // RECORD.itemsize
        if not n:
            continue
        data = np.memmap(p, mode="r", dtype=RECORD, shape=(n,))
        if int(data[-1]["mono"]) < start-250_000_000 or int(data[0]["mono"]) > end+100_000_000:
            continue
        low, high = data["mono"].searchsorted([start-250_000_000, end+100_000_000])
        chunk = data[low:high]
        chunks.append(chunk[chunk["port"] == 0].copy())
        matching.append(p)
    if not chunks:
        return [], {}
    padded = np.concatenate(chunks)
    within = (padded["mono"] >= start) & (padded["mono"] <= end)
    error_count = int(np.sum((padded["cid"][within] & 0x20000000) != 0))
    status = padded["data"][within & (padded["cid"] == 0x2a1)]
    faults = sum(bool(d.tobytes()[1] or d.tobytes()[6:8] != bytes(2)) for d in status)
    hold = []
    for axis in range(6):
        cmd = padded[(padded["cid"] == 0x155+axis//2) & (padded["size"] == 8)]
        fb = padded[(padded["cid"] == 0x2a5+axis//2) & (padded["size"] == 8)]
        if len(cmd) == 0 or len(fb) == 0:
            hold.append(None)
            continue
        def positions(data):
            return np.frombuffer(data["data"].tobytes(), dtype=">i4").reshape(-1, 2)[:,axis % 2]
        changed = cmd["wall"].astype("i8")[np.r_[True, np.diff(positions(cmd)) != 0]]
        times = fb["wall"].astype("i8")
        earlier = times.searchsorted(times - 20_000_000)
        latest = changed.searchsorted(times, side="right") - 1
        sel = (latest >= 0) & (times - changed[latest] >= 150_000_000) & (times-times[earlier] >= 10_000_000)
        sel &= (fb["mono"] >= start) & (fb["mono"] <= end)
        step = abs(positions(fb).astype(float) - positions(fb)[earlier]) / 1000
        hold.append(dict(tested=int(sel.sum()), max_deg=round(float(step[sel].max()), 3) if sel.any() else None,
                         flags=int(np.sum(sel & (step >= .7)))))
    return matching, dict(errors=error_count, status_frames=len(status), status_faults=faults, hold=hold)


def audit(base, name):
    episode = base / "recordings" / name
    meta = json.loads((episode / "episode_summary.json").read_text())
    rows = [json.loads(line) for line in (episode / "alignment.jsonl").read_text().splitlines()]
    st = np.asarray([r["sample_monotonic_ns"] for r in rows], dtype="i8")
    keys = [f"{Decimal(int(t))/Decimal(10**9):.6f}" for t in st]
    report = dict(rows=len(rows), unique=len(set(keys)), gap_ms=stats(np.diff(st)/1e6),
                  duration_s=round((st[-1]-st[0])/1e9, 3), source=meta.get("action_source"),
                  skew_ms=stats([abs(r["cameras"]["overhead"]["monotonic_ns"]-r["cameras"]["wrist"]["monotonic_ns"])/1e6 for r in rows]),
                  streams=inspect_streams(episode, rows, keys))
    for role in ("target", "feedback"):
        valid = [i for i,r in enumerate(rows) if r[role] is not None]
        ages = [(st[i]-rows[i][role]["sample_monotonic_ns"])/1e6 for i in valid]
        report[role] = dict(valid=len(valid), leading=valid[0] if valid else len(rows),
                            missing_after=len(rows)-valid[0]-len(valid) if valid else 0,
                            age_ms=stats(ages) if valid else None)
    files = sorted((base.parent/"diagnostics").glob("piper_passive_*/frames-*.bin"))
    matching, report["passive"] = passive_window(files, int(st[0]), int(st[-1]))
    try:
        report["independent"] = check(episode, matching)
    except ValueError as e:
        report["independent"] = dict(error=str(e))
    report["endpoint"] = endpoints(episode)
    width = np.asarray([r["feedback"]["gripper_mm"] for r in rows if r["feedback"] is not None])
    if len(width):
        report["gripper_feedback_mm"] = dict(first=round(float(width[0]),2), min=round(float(width.min()),2),
                                             last=round(float(width[-1]),2))
    vendor = (episode/"data_tools.log").read_text(errors="replace").lower()
    report["log_terms"] = {term: vendor.count(term) for term in ("timeout", "disconnect", "traceback", "error parsing json")}
    if meta.get("outcome") == "aborted":
        report["camera_log"] = camera_log(episode)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1]),sys.argv[2])))
