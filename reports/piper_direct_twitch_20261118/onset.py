"""Compare feedback near the onsite marker against the preceding stationary window."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
import argparse
import socket
import struct

ROW = struct.Struct('<QQIBBH8s')
SECOND = 1_000_000_000


def records(directory, start, end):
    for path in sorted(directory.glob('frames-*.bin')):
        size = path.stat().st_size // ROW.size * ROW.size
        if not size:
            continue
        with path.open('rb') as f:
            f.seek(size - ROW.size)
            if ROW.unpack(f.read(ROW.size))[0] < start:
                continue
            f.seek(0)
            while f.tell() < size:
                chunk = f.read(min(2**20, size-f.tell()))
                for row in ROW.iter_unpack(chunk):
                    if start <= row[0] < end and row[3] == 0:
                        yield row


def main(directory, baseline_start, search_start, end):
    lo, split, hi = [int(x.timestamp()) * SECOND for x in (baseline_start,search_start,end)]
    tz = search_start.tzinfo

    def stamp(t):
        return datetime.fromtimestamp(t//SECOND,tz).strftime('%H:%M:%S') + f'.{(t%SECOND)//1000:06d}'

    channels = defaultdict(list)
    counts = Counter()
    statuses = Counter()
    tx = errors = 0
    for ns, _, cid, _, length, flags, data in records(directory, lo, hi):
        counts[cid] += 1
        errors += bool(cid & 0x20000000)
        tx += bool(flags & socket.MSG_DONTROUTE)
        if length != 8:
            continue
        if cid == 0x2a1:
            statuses[data.hex()] += 1
        elif 0x2a5 <= cid <= 0x2a7:
            for j, value in enumerate(struct.unpack('>ii',data),1+(cid-0x2a5)*2):
                channels[j,'position_deg'].append((ns,value/1000))
        elif 0x251 <= cid <= 0x256:
            speed,current,_ = struct.unpack('>hhi',data)
            channels[cid-0x250,'speed_rad_s'].append((ns,speed/1000))
            channels[cid-0x250,'current_A'].append((ns,current/1000))
    print('COUNTS',sum(counts.values()),'ids',{hex(k):n for k,n in sorted(counts.items())},
          'local_tx',tx,'error_frames',errors,'arm_status',dict(statuses))
    # The thresholds define a reproducible observation, independent of any firmware stop thresholds.
    floors = {'position_deg':0.2, 'speed_rad_s':0.5, 'current_A':0.5}
    for key, values in sorted(channels.items()):
        values.sort()
        baseline=[v for ns,v in values if ns<split]
        observations=[(ns,v) for ns,v in values if ns>=split]
        if not baseline or not observations:
            continue
        axis,field=key
        ref=median(baseline)
        if field=='position_deg':
            print('REFERENCE',axis,field,min(baseline),max(baseline),len(baseline))
        for scale in (0.5,1.0,2.0):
            margin=max(floors[field]*scale,max(abs(v-ref) for v in baseline)+0.001)
            indices=[i for i,(_,v) in enumerate(observations) if abs(v-ref)>=margin]
            if indices:
                i=indices[0]
                t,v=observations[i]
                prev=observations[i-1] if i else [(ns,v) for ns,v in values if ns<split][-1]
                limit=7_500_000 if field=='position_deg' else 15_000_000
                sustained=next(((ns,x) for (ns,x),(nt,y) in zip(observations,observations[1:])
                                if nt-ns<=limit and abs(x-ref)>=margin and abs(y-ref)>=margin),None)
                print('CROSS',axis,field,'scale',scale,'baseline',ref,'range',[min(baseline),max(baseline)],
                      'margin',round(margin,4),'previous',(stamp(prev[0]),prev[1]),'first',(stamp(t),v),
                      'two_sample_start',(stamp(sustained[0]),sustained[1]) if sustained else None)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    for name in ('baseline_start','search_start','end'):
        ap.add_argument(name,type=datetime.fromisoformat)
    args=ap.parse_args()
    dates=(args.baseline_start,args.search_start,args.end)
    if any(x.utcoffset() is None or x.microsecond for x in dates) or not dates[0]<dates[1]<dates[2]:
        ap.error('Provide three increasing whole-second timestamps with UTC offsets.')
    main(args.directory,*dates)
