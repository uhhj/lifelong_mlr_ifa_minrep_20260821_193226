"""Summarize the direct-link incident from existing watch files, without opening CAN."""
from collections import Counter, defaultdict
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
import argparse
import json
import struct

ROW = struct.Struct('<QQIBBH8s')


def stamp(ns, tz):
    return datetime.fromtimestamp(ns / 1e9, tz).isoformat(timespec='milliseconds')


def main(watch, start, end):
    lo, hi = (int(x.timestamp()) * 10**9 for x in (start, end))
    pre = lo - 120 * 10**9
    counts = Counter()
    pre_counts = Counter()
    statuses = defaultdict(list)
    positions = defaultdict(list)
    drives = defaultdict(list)
    speeds = defaultdict(list)
    first = last = None
    for path in sorted(watch.glob('frames-*.bin')):
        size = path.stat().st_size // ROW.size * ROW.size
        if not size:
            continue
        with path.open('rb') as f:
            f.seek(size - ROW.size)
            if ROW.unpack(f.read(ROW.size))[0] < pre:
                continue
            f.seek(0)
            while f.tell() < size:
                chunk = f.read(min(2**20, size - f.tell()))
                for ts, _, cid, port, n, flags, data in ROW.iter_unpack(chunk):
                    if not pre <= ts < hi:
                        continue
                    if lo <= ts:
                        counts[(port,cid)] += 1
                        continue
                    first = ts if first is None else min(first,ts)
                    last = ts if last is None else max(last,ts)
                    pre_counts[(port,cid)] += 1
                    if n != 8 or port != 0:
                        continue
                    if cid == 0x2a1:
                        statuses[data.hex()].append(ts)
                    elif 0x2a5 <= cid <= 0x2a7:
                        for i,v in enumerate(struct.unpack('>ii',data)):
                            positions[1+2*(cid-0x2a5)+i].append((ts,v))
                    elif 0x261 <= cid <= 0x266:
                        drives[cid-0x260].append((ts,*struct.unpack('>HhbBH',data)))
                    elif 0x251 <= cid <= 0x256:
                        speeds[cid-0x250].append((ts,*struct.unpack('>hhi',data)))

    health = []
    relevant = []
    with (watch/'health.jsonl').open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            ts = row['wall_ns']
            if pre<=ts<hi and row['kind'] != 'health':
                relevant.append([stamp(ts,start.tzinfo), row['kind'], row.get('interface'), row.get('errno')])
            if lo<=ts<hi and row['kind']=='health':
                health.append((ts,tuple(row['bound']),{k:v.get('frames',0) for k,v in row['counts'].items()}))

    tz=start.tzinfo
    print('WINDOW',start.isoformat(),end.isoformat(),'total_frames',sum(counts.values()),'health_rows',len(health))
    if health:
        print('HEALTH',stamp(health[0][0],tz),stamp(health[-1][0],tz),'bound',dict(Counter(str(x[1]) for x in health)),
              'max_gap_ms',round(max((b[0]-a[0] for a,b in zip(health,health[1:])),default=0)/1e6,2),
              'counts_first_last',health[0][2],health[-1][2])
    print('EVENTS',relevant)
    if last is None:
        return
    print('PRE',stamp(first,tz),stamp(last,tz),'count',sum(pre_counts.values()),
          'commands',sum(n for (_,cid),n in pre_counts.items() if cid in (0x151,0x155,0x156,0x157,0x159)),
          'error_frames',sum(n for (_,cid),n in pre_counts.items() if cid&0x20000000))
    print('STATUS',{k:[len(v),stamp(min(v),tz),stamp(max(v),tz)] for k,v in statuses.items()})
    for j,ser in sorted(positions.items()):
        _,q0=ser[0]
        excursions=[(t,x) for t,x in ser if abs(x-q0)>=200]
        low=drives[j]
        high=speeds[j]
        print('AXIS',j,'first_last_deg',[q0/1000,ser[-1][1]/1000],
              'range_deg',(max(x for t,x in ser)-min(x for t,x in ser))/1000,
              'first_offset_0.2_deg', stamp(excursions[0][0],tz) if excursions else None,
              'last_position_time',stamp(ser[-1][0],tz))
        timestamps=[t for t,_ in ser]
        best=None
        for i,(t2,x2) in enumerate(ser):
            k=bisect_left(timestamps,t2-20_000_000,0,i+1)
            t1,x1=ser[k]
            if i-k>=2 and (best is None or abs(x2-x1)>abs(best[-1])):
                best=(t1,t2,x1,x2,x2-x1)
        if best is not None and high:
            t1,t2,x1,x2,delta=best
            h=[v for v in high if t1<=v[0]<=t2]
            print('FAST',j,stamp(t1,tz),stamp(t2,tz),'position_deg',[x1/1000,x2/1000],
                  'delta_deg',delta/1000,'dt_ms',round((t2-t1)/1e6,3),'motor_speed_current',
                  [[stamp(v[0],tz),v[1]/1000,v[2]/1000] for v in h])
        if low:
            print('DRIVE',j,'voltage_V',sorted({u[1]/10 for u in low}),
                  'status_counts',dict(Counter(hex(u[4]) for u in low)),'last_time',stamp(low[-1][0],tz))
        if high:
            print('MOTOR',j,'speed_min_max_rad_s',[fn(x[1] for x in high)/1000 for fn in (min,max)],
                  'current_min_max_A',[fn(x[2] for x in high)/1000 for fn in (min,max)],'last_time',stamp(high[-1][0],tz))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('watch',type=Path)
    ap.add_argument('start',type=datetime.fromisoformat)
    ap.add_argument('end',type=datetime.fromisoformat)
    args=ap.parse_args()
    if not all(d.utcoffset() is not None and d.microsecond==0 for d in (args.start,args.end)) or args.end<=args.start:
        ap.error('Use increasing timestamps with offsets and whole-second boundaries.')
    main(args.watch,args.start,args.end)
