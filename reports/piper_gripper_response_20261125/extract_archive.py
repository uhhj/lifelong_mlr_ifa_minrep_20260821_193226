"""Find isolated locally generated width commands in retained binary CAN captures."""

import bisect
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import mmap
import re
from pathlib import Path
from struct import Struct, pack, unpack_from
import sys

w = Path(sys.argv[1])
tz = timezone(timedelta(hours=8))
stamp = lambda t: datetime.fromtimestamp(t / 1e9, tz).isoformat(timespec="milliseconds")
rec = Struct("<QQIBBH8s")
decode = Struct(">ihBB")
all_parts = []
sources = Counter()
tx = []
exec_windows = []
for path in (w / "policy_runs").glob("2026112*.jsonl"):
    if not path.name.startswith(("20261121", "20261122")):
        continue
    with path.open("rb") as f:
        h = json.loads(f.readline())
        off = h["wall_ns"] - h["mono_ns"]
        begin = None
        for line in f:
            if b'"start"' not in line and b'"end"' not in line:
                continue
            q = json.loads(line)
            if q["event"] == "start":
                begin = q["mono_ns"] + off
            if q["event"] == "end" and begin is not None:
                exec_windows.append((begin, q["mono_ns"] + off))
needle = re.compile(re.escape(pack("<IBB", 0x159, 0, 8)))
for path in sorted((w.parent / "diagnostics").glob("piper_passive_20261121*/frames-*.bin")):
    with path.open("rb") as f:
        n = path.stat().st_size // rec.size
        if not n:
            continue
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            lo = unpack_from("<Q", m, 0)[0]
            hi = unpack_from("<Q", m, (n - 1) * rec.size)[0]
            all_parts.append((lo, hi, path))
            for hit in needle.finditer(m):
                start = hit.start() - 16
                if start < 0 or start % rec.size:
                    continue
                wall, _, _, _, _, flag, data = rec.unpack_from(m, start)
                val, eff, code, zero = decode.unpack(data)
                sources[(bool(flag & 4), eff, code, zero)] += 1
                if flag & 4:
                    tx.append((wall, val, eff, code, zero))
tx.sort()


def window(a, b):
    out = []
    for lo, hi, path in all_parts:
        if lo > b or hi < a:
            continue
        with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            ix = range(len(m) // rec.size)
            timestamp = lambda i: unpack_from("<Q", m, i * rec.size)[0]
            first = bisect.bisect_left(ix, a, key=timestamp)
            last = bisect.bisect_right(ix, b, key=timestamp)
            for wall, _, cid, port, _, flags, data in rec.iter_unpack(m[first * rec.size : last * rec.size]):
                if port == 0 and (cid in (0x2A8, 0x2A1, 0x155, 0x159, 0x150, 0x471) or cid & 0x20000000):
                    out.append((wall, cid, flags, data))
    return sorted(out, key=lambda v: v[0])


events = []
for i, (t, val, eff, code, zero) in enumerate(tx):
    if val % 1000 or not 0 <= val <= 70000 or (eff, code, zero) != (1000, 1, 0):
        continue
    if any(a - 500_000_000 <= t <= b + 500_000_000 for a, b in exec_windows):
        continue
    if i and t - tx[i - 1][0] < 200_000_000:
        continue
    if i + 1 < len(tx) and tx[i + 1][0] - t < 500_000_000:
        continue
    frames = window(t - 200_000_000, t + 500_000_000)
    fb = [(u, decode.unpack(data)) for u, cid, flags, data in frames if cid == 0x2A8 and not flags & 4]
    before = [v for u, v in fb if u < t]
    after = [(u, v) for u, v in fb if u >= t]
    if len(before) < 10 or len(after) < 25 or after[-1][0] < t + 470_000_000:
        continue
    if min(v[2] for _, v in fb) != 64 or max(v[2] for _, v in fb) != 64:
        continue
    if any(cid & 0x20000000 for _, cid, _, _ in frames):
        continue
    statuses = [data.hex() for _, cid, flags, data in frames if cid == 0x2A1 and not flags & 4]
    w0 = before[-1][0] / 1000
    orient = 1 if val / 1000 > w0 else -1
    reach = [(u - t) / 1e6 for u, v in after if orient * (v[0] / 1000 - w0) >= 0.3 - 1e-9]
    limits = lambda a: [min(a), max(a)]
    events.append(
        dict(
            time=stamp(t),
            target_mm=val / 1000,
            initial_mm=w0,
            range_before=limits([v[0] / 1000 for v in before]),
            final_500ms=after[-1][1][0] / 1000,
            first_move_0p3_ms=reach[0] if reach else None,
            torque_range=limits([v[1] / 1000 for _, v in after]),
            raw_status=sorted(set(statuses))[:3],
            joint_targets=sum(cid == 0x155 for _, cid, _, _ in frames),
            n_feedback=len(after),
        )
    )
bursts = []
for row in tx:
    if not bursts or row[0] - bursts[-1][-1][0] > 500_000_000:
        bursts.append([])
    bursts[-1].append(row)
series = []
for group in bursts:
    start, end = group[0][0], group[-1][0]
    frames = window(start - 350_000_000, end + 450_000_000)
    feedback = []
    status = []
    safety = []
    for u, cid, flags, data in frames:
        if cid == 0x2A8 and not flags & 4:
            v, eff, code, _ = decode.unpack(data)
            feedback.append([u, v, eff, code])
        if cid == 0x2A1 and not flags & 4 and (not status or status[-1][1] != data.hex()):
            status.append([u, data.hex()])
        if cid in (0x150, 0x471) or cid & 0x20000000:
            safety.append([u, hex(cid), data.hex()])
    series.append(
        dict(
            start=start,
            end=end,
            commands=group,
            feedback=feedback,
            states=status,
            safety=safety,
            overlaps_policy=any(a <= end and b >= start for a, b in exec_windows),
        )
    )
print(
    json.dumps(
        dict(
            files=len(all_parts),
            hours=sum(b - a for a, b, _ in all_parts) / 3.6e12,
            earliest=stamp(min(a for a, _, _ in all_parts)),
            latest=stamp(max(b for _, b, _ in all_parts)),
            candidate_tx=len(tx),
            common_headers=[(str(k), v) for k, v in sources.most_common(8)],
            isolated=events,
            series=series,
        )
    )
)
