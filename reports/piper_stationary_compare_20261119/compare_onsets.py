"""Offline comparison of joint and drive CAN feedback on the receiver clock."""
from datetime import datetime, timezone, timedelta
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TZ = timezone(timedelta(hours=8))
NS = 1_000_000_000
DT = np.dtype([('wall','<u8'),('mono','<u8'),('cid','<u4'),('port','u1'),
               ('dlc','u1'),('flags','<u2'),('data','V8')])


def ns(time):
    return int(datetime.fromisoformat(time).replace(tzinfo=TZ).timestamp()) * NS


def clock(t):
    sec, nano = divmod(int(t), NS)
    return datetime.fromtimestamp(sec,TZ).strftime('%H:%M:%S') + f'.{nano//1000:06}'


def load_window(pattern, lo, hi):
    rows = []
    for name in sorted(glob.glob(pattern)):
        p = Path(name)
        n = p.stat().st_size // DT.itemsize
        if not n:continue
        a = np.memmap(p,mode='r',dtype=DT,shape=(n,))
        t = a['wall']
        if int(t[n-1]) < lo or int(t[0]) >= hi:continue
        left,right = t.searchsorted(np.array([lo,hi],dtype='<u8'))
        if right>left:
            rows.append(np.array(a[left:right]))
    if not rows:raise RuntimeError('Missing source window: ' + pattern)
    d=np.concatenate(rows)
    d=d[d['port']==0]
    if np.any(np.diff(d['wall'].astype('i8'))<0):raise ValueError('non-monotone kernel receive time')
    return d


def channels(d):
    result={}
    for axis in range(6):
        pos=d[(d['cid']==0x2a5+axis//2)&(d['dlc']==8)]
        high=d[(d['cid']==0x251+axis)&(d['dlc']==8)]
        assert len(pos) and len(high)
        pos_values=np.frombuffer(pos['data'].tobytes(),dtype='>i4').reshape(-1,2)
        hv=np.frombuffer(high['data'].tobytes(),dtype='>i2').reshape(-1,4)
        motor_position=np.frombuffer(high['data'].tobytes(),dtype='>i4').reshape(-1,2)[:,1]
        result[axis+1]={
            'q': (pos['wall'].astype('i8'),pos_values[:,axis%2].astype(float)/1000),
            'v': (high['wall'].astype('i8'),hv[:,0].astype(float)/1000),
            'i': (high['wall'].astype('i8'),hv[:,1].astype(float)/1000),
            'motor_raw': (high['wall'].astype('i8'),motor_position)}
    return result


def profile(ch,lo,hi):
    previous=None
    for t in range(lo,hi,NS):
        stats=[]
        for axis in ch:
            qt,q=ch[axis]['q']; vt,v=ch[axis]['v']
            a,b=qt.searchsorted([t,t+NS]); c,d=vt.searchsorted([t,t+NS])
            if b-a and d-c:stats.append((np.ptp(q[a:b]),np.max(abs(v[c:d]))))
        if len(stats)!=6:continue
        span,speed=np.max(stats,axis=0)
        quiet=span<.06 and speed<.12
        if quiet!=previous or not quiet and t>=hi-30*NS:
            print('PROFILE',clock(t),quiet,'q_span',round(span,3),'max_v',round(speed,3))
        previous=quiet


def examine(d, base_start, split, stop, detail=True):
    ch=channels(d)
    counts={hex(k):int(np.count_nonzero(d['cid']==k)) for k in (0x150,0x151,0x155,0x156,0x157,0x159,0x470,0x471)}
    print('SCOPE',clock(d['wall'][0]),clock(d['wall'][-1]),len(d),'sha256_window',hashlib.sha256(d.tobytes()).hexdigest(),
          'control',counts,'local_echo',np.count_nonzero(d['flags']&4),'errors',np.count_nonzero(d['cid']&0x20000000))
    profile(ch,base_start,stop)
    if not detail:return
    print('BASE',clock(base_start),clock(split),'SEARCH_END',clock(stop))
    scales=(.5,1.,2.)
    for axis,z in ch.items():
        for field,floor in (('q',.2),('v',.5),('i',.5)):
            t,x=z[field]; lo,mid,end=t.searchsorted([base_start,split,stop])
            ref=np.median(x[lo:mid]); tolerance=np.max(abs(x[lo:mid]-ref))+.001
            delta=abs(x[mid:end]-ref)
            sampling=np.diff(t[lo:end])/1e6
            bounds=[round(float(a),4) for a in (ref,x[lo:mid].min(),x[lo:mid].max())]
            print('BASELINE',axis,field,bounds,'median_max_gap_ms',[round(float(a),3) for a in (np.median(sampling),np.max(sampling))])
            for s in scales:
                margin=max(floor*s,tolerance)
                # Decimal device units; include an exactly-on-threshold measurement.
                crossed=delta+1e-10>=margin
                first=np.flatnonzero(crossed)
                if len(first)==0:continue
                k=mid+first[0]
                sustained=np.flatnonzero(crossed[:-1]&crossed[1:]&(np.diff(t[mid:end])<=(7_500_000 if field=='q' else 15_000_000)))
                r=clock(t[mid+sustained[0]]) if len(sustained) else None
                print('HIT',s,axis,field,round(margin,5),clock(t[k-1]),round(x[k-1],5),clock(t[k]),round(x[k],5),'two',r)
        t,x=z['q']; mask=(t>=split)&(t<stop)
        left=t.searchsorted(t-20_000_000)
        ids=np.flatnonzero(mask&(t-t[left]>=9_000_000)&(abs(x-x[left])+.0000000001>=.7))
        if len(ids):
            k=ids[0]; print('RAPID',axis,clock(t[left[k]]),clock(t[k]),round(x[k]-x[left[k]],4))
    near=[]
    for axis,z in ch.items():
        t,x=z['q']; a,b,c=t.searchsorted([base_start,split,stop])
        ref=np.median(x[a:b])
        changed=np.flatnonzero(abs(x[b:c]-ref)+1e-10>=.1)
        if len(changed):near.append((int(t[b+changed[0]]),axis))
    if near:
        origin=min(near)[0]
        for t_hit,axis in sorted(near):
            if t_hit>origin+25_000_000:continue
            z=ch[axis]; th,v=z['v']; amps=z['i'][1]; motor=z['motor_raw'][1]
            tq,q=z['q']
            rows=[(clock(t),round(float(x),3)) for t,x in zip(tq,q) if origin-15_000_000<=t<=origin+35_000_000]
            drive=[(clock(th[k]),round(v[k],3),round(amps[k],3),int(motor[k]))
                   for k in np.flatnonzero((th>=origin-22_000_000)&(th<=origin+35_000_000))]
            print('EDGE',axis,'q',rows,'drive_v_i_raw_position',drive)
    # Low-speed driver feedback carries bus voltage (0.1 V) and FOC status.
    for axis in ch:
        low=d[(d['cid']==0x260+axis)&(d['dlc']==8)]
        if len(low):
            words=np.frombuffer(low['data'].tobytes(),dtype='>u2').reshape(-1,4)
            octets=np.frombuffer(low['data'].tobytes(),dtype='u1').reshape(-1,8)
            print('DRIVE_RANGE',axis,'volts',float(min(words[:,0])/10),float(max(words[:,0])/10),
                  'status_values',list(map(int,np.unique(octets[:,5]))))
    full=d[(d['cid']==0x2a1)&(d['dlc']==8)]
    print('ARM_STATUS',[(x.tobytes().hex(),int(n)) for x,n in zip(*np.unique(full['data'],return_counts=True))])


if __name__=='__main__':
    # Each input supplies a segment glob, baseline start, search start and search end.
    for name,pattern,start,split,stop in json.loads(sys.argv[1]):
        a,b,c=map(ns,(start,split,stop))
        print('EVENT',name)
        examine(load_window(pattern,a,c),a,b,c)
