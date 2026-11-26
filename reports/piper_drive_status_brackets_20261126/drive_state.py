"""Bracket a field event with fresh per-axis drive-status frames from archived can0 logs."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timedelta, timezone
import glob
import json
from pathlib import Path
import struct

ROW = struct.Struct("<QQIBBH8s")
TZ = timezone(timedelta(hours=8))


def report(pattern, at):
    moment = datetime.fromisoformat(at)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TZ)
    center = int(moment.timestamp()) * 10**9 + moment.microsecond * 1000
    axis_rows = [[] for _ in range(6)]
    traffic = Counter()
    for name in glob.glob(pattern):
        with Path(name).open("rb") as file:
            rows = Path(name).stat().st_size // ROW.size

            def stamp(i):
                file.seek(i * ROW.size)
                return int.from_bytes(file.read(8), "little")

            if not rows or stamp(0) > center + 1_000_000_000 or stamp(rows-1) < center - 1_000_000_000:
                continue
            a = bisect_left(range(rows), center-1_000_000_000, key=stamp)
            b = bisect_right(range(rows), center+1_000_000_000, key=stamp)
            file.seek(a * ROW.size)
            for wall, _, cid, port, dlc, flags, payload in ROW.iter_unpack(file.read((b-a)*ROW.size)):
                if port != 0:
                    continue
                if cid & 0x20000000:
                    traffic["error_frames"] += 1
                if cid in (0x150,0x151,0x155,0x156,0x157,0x159,0x470,0x471):
                    traffic[hex(cid)] += 1
                if 0x261 <= cid <= 0x266 and dlc == 8 and not flags & (4|8|32):
                    axis_rows[cid-0x261].append((wall,payload[5]))
    axes=[]
    for j, rows in enumerate(axis_rows,1):
        rows.sort()
        before = [r for r in rows if r[0] <= center]
        after = [r for r in rows if r[0] > center]
        def sample(r):
            return {"offset_ms": round((r[0]-center)/1e6,3), "status": hex(r[1]), "enabled":bool(r[1]&0x40)}
        axes.append({"axis":j, "before":sample(before[-1]) if before else None,
                     "after":sample(after[0]) if after else None,"n":len(rows),
                     "interval_statuses": sorted({hex(r[1]) for r in rows})})
    return {"event":moment.isoformat(timespec="microseconds"),"axes":axes,
            "command_and_error_counts_within_one_second_each_side":dict(traffic)}


if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("file_pattern")
    cli.add_argument("time_beijing")
    args=cli.parse_args()
    print(json.dumps(report(args.file_pattern,args.time_beijing)))
