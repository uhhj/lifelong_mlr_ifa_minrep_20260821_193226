"""Align a completed PiPER run with receive-only SocketCAN records."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import socket
import struct
import sys

import numpy as np


IDS = (0x155, 0x156, 0x157)
FB = (0x2A5, 0x2A6, 0x2A7)
ROW = np.dtype([('wall','<u8'),('mono','<u8'),('id','<u4'),('port','u1'),('dlc','u1'),('flags','<u2'),('data','S8')])
ERROR = 0x20000000
NS = 1_000_000_000


def wall(t):
    s, extended = divmod(int(t), NS)
    return datetime.fromtimestamp(s).astimezone().strftime('%H:%M:%S')+f'.{extended:09d}'


def show(label, **data):
    print(label, json.dumps(data, separators=(',', ':'), default=lambda x: x.item()))


def packet_events(watch, start, end):
    selected = []
    for path in sorted(watch.glob('frames-*.bin')):
        count = path.stat().st_size // ROW.itemsize
        if not count:
            continue
        log = np.memmap(path, mode='r', dtype=ROW, shape=(count,))
        if log[-1]['wall'] + NS < start or log[0]['wall'] > end + NS:
            continue
        selected.append(log[(log['wall'] >= start) & (log['wall'] <= end)])
    events = np.concatenate(selected)
    return events[np.argsort(events['wall'],kind='stable')]


def rows(events, port, cid, is_tx=None):
    keep = (events['port']==port)&(events['id']==cid)
    if is_tx is not None:
        tx = (events['flags'] & socket.MSG_DONTROUTE) != 0
        keep &= tx==is_tx
    return events[keep]


def decode(r, layout):
    return struct.unpack(layout, r['data'].tobytes().ljust(8,b'\0'))


def series(records, layout):
    return records['wall'].astype(np.int64), np.array([decode(r,layout) for r in records])


def groups(events, port, ids, is_tx, maximum_ns):
    mask = (events['port']==port) & np.isin(events['id'],ids)
    if is_tx is not None:
        mask &= (((events['flags']&socket.MSG_DONTROUTE)!=0)==is_tx)
    result = []
    pending = []
    for row in events[mask]:
        cid = int(row['id']); stamp = int(row['wall'])
        if cid==ids[0]:
            pending=[row]
        elif pending and cid==ids[len(pending)] and stamp-int(pending[0]['wall'])<=maximum_ns:
            pending.append(row)
            if len(pending)==3:
                result.append((stamp,[v for r in pending for v in decode(r,'>ii')]))
                pending=[]
        else:
            pending=[]
    return np.array([t for t,_ in result]), np.array([q for _,q in result])


def peaks(t, q, targets, from_ns, to_ns):
    tt,tq=targets
    left=np.searchsorted(t,t-20_000_000)
    t_left=np.searchsorted(tt,t-120_000_000,side='right')-1
    t_right=np.searchsorted(tt,t,side='right')
    ordered=[]
    for i in np.flatnonzero((t>=from_ns)&(t<=to_ns)&(t_left>=0)&(left<np.arange(len(t)))):
        target=tq[t_left[i]:t_right[i]]
        delta=q[i]-q[left[i]]
        for j in np.flatnonzero((np.ptp(target,axis=0)<=100)&(abs(delta)>=700)):
            ordered.append((int(abs(delta[j])),int(t[i]),int(j),int(delta[j]),int(left[i]),int(i)))
    retained=[]
    for magnitude, stamp, j, signed, old, new in sorted(ordered,reverse=True):
        if not any(axis==j and abs(when-stamp)<100_000_000 for _,when,axis,*_ in retained):
            retained.append((magnitude,stamp,j,signed,old,new))
    return retained


def window(t, y, ns, before=65, after=35):
    keep = (t >= ns-before*1_000_000)&(t <= ns+after*1_000_000)
    return [[round((x-ns)/1e6,3),*p.tolist()] for x,p in zip(t[keep],y[keep])]


def main(run, watch, episode):
    samples=[json.loads(line) for line in (run/'samples.jsonl').open()]
    anchors=[r['wall']-r['mono'] for r in samples]
    offset=int(float(np.median(anchors))*NS)
    raw_start=int(round(samples[0]['mono']*NS))+offset-3*NS
    raw_end=(run/'result.json').stat().st_mtime_ns+NS
    alignment=[json.loads(line) for line in (episode/'alignment.jsonl').open()]
    clip_start=alignment[0]['sample_monotonic_ns']+offset
    clip_end=alignment[-1]['sample_monotonic_ns']+offset
    captured=packet_events(watch,raw_start,raw_end)
    err=captured[(captured['id']&ERROR)!=0]
    frames=Counter((int(x['port']),int(x['id']),bool(int(x['flags'])&socket.MSG_DONTROUTE)) for x in captured)
    show('COVERAGE', session=[wall(raw_start),wall(raw_end)], recording=[wall(clip_start),wall(clip_end)],
         captured=len(captured), errors=[(int(e['port']),wall(e['wall']),hex(int(e['id']&0x1FFFFFFF))) for e in err],
         adj_wall_offset_ns=offset, frames={f'{p}:{c:03x}:{"TX" if tx else "RX"}':n for (p,c,tx),n in sorted(frames.items())})
    t_m,q_m=groups(captured,1,IDS,False,3_000_000)
    t_t,q_t=groups(captured,0,IDS,True,12_000_000)
    t_f,q_f=groups(captured,0,FB,False,25_000_000)
    high={j:series(rows(captured,0,0x251+j,False),'>hhi') for j in range(6)}
    low={j:series(rows(captured,0,0x261+j,False),'>HhbBH') for j in range(6)}
    g_t,g_v=series(rows(captured,0,0x159,True),'>iHBB')
    stop=rows(captured,0,0x150,True)
    show('STREAMS', leader=len(t_m), echo=len(t_t), position=len(t_f), stop=[wall(t) for t in stop['wall']])
    if len(t_m)+1 == len(t_t):
        show('MATCH', exact=int(np.sum(np.all(q_m == q_t[1:],axis=1))), groups=len(t_m),
             max_forward_ms=float(np.max(t_t[1:]-t_m)/1e6))
    for j in range(6):
        t,h=high[j]
        show('HIGH',joint=j+1,frames=len(t),dt_ms=list(np.round(np.percentile(np.diff(t)/1e6,(0,50,95,99,100)),4)),
             speed_bounds=h[:,0].min().item(),speed_max=h[:,0].max().item(),current_min=h[:,1].min().item(),current_max=h[:,1].max().item(),
             encoder_min=h[:,2].min().item(),encoder_max=h[:,2].max().item())
    maximum=stop['wall'][0] if len(stop) else raw_end
    candidates=peaks(t_f,q_f,(t_t,q_t),t_t[1],min(maximum,t_t[-1]))
    for k,(value,t,j,signed,old,new) in enumerate(candidates[:24]):
        show('CANDIDATE', rank=k+1, time=wall(t), at_monotonic_ns=t-offset,joint=j+1,
             begin=wall(t_f[old]), delta_mdeg=signed, prior=q_f[old,j], position=q_f[new,j],
             from_clip_s=round((t-clip_start)/NS,4), in_recording=clip_start<=t<=clip_end)
    for magnitude,t,j,signed,i0,i1 in candidates[:10]:
        position_time, position_matrix = series(rows(captured,0,FB[j//2],False),'>ii')
        p=position_matrix[:,j%2:j%2+1]
        m_next=np.searchsorted(t_m,t,side='right')
        t_next=np.searchsorted(t_t,t,side='right')
        show('DETAIL', stamp=int(t),time=wall(t), joint=j+1,
             meas=window(position_time,p,t), motor=window(*high[j],t), drive=window(*low[j],t,110,35),
             surrounding_leader=[wall(x) for x in t_m[max(0,m_next-1):m_next+1]],
             surrounding_echo=[wall(x) for x in t_t[max(0,t_next-1):t_next+1]],
             target=window(t_t,q_t,t), master=window(t_m,q_m,t), grip=window(g_t,g_v,t),
             other_motors={a+1:window(*high[a],t,35,20) for a in range(6) if a!=j})


if __name__ == '__main__':
    main(*map(Path,sys.argv[1:]))
