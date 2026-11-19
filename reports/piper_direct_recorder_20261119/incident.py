"""Align the onsite 15:57 incident to frozen receive-only frames."""
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
import struct
import numpy as np

DT=np.dtype([('wall','<u8'),('mono','<u8'),('cid','<u4'),('port','u1'),('dlc','u1'),('flags','<u2'),('data','V8')])


def clock(ns):
    return datetime.fromtimestamp(int(ns)//10**9).astimezone().strftime('%H:%M:%S')+f'.{int(ns)%10**9//1000:06}'


def grouped(events,base):
    mask=(events['cid']>=base)&(events['cid']<=base+2)
    s=events[mask]
    ids=s['cid'];t=s['wall'].astype('int64')
    good=np.flatnonzero((ids[:-2]==base)&(ids[1:-1]==base+1)&(ids[2:]==base+2)&(t[2:]-t[:-2]<=3_000_000))
    q=np.frombuffer(s['data'].tobytes(),dtype='>i4').reshape(-1,2)
    return t[good+2],np.column_stack((q[good],q[good+1],q[good+2]))


def audit(path):
    data=np.memmap(path,dtype=DT,mode='r')
    start=int(datetime.fromisoformat('2026-11-19T15:56:30+08:00').timestamp())*10**9
    events=data[(data['wall']>=start)&(data['port']==0)]
    tt,cmd=grouped(data[data['port']==0],0x155)
    ft,fb=grouped(events,0x2a5)
    print('COVER',clock(events['wall'][0]),clock(events['wall'][-1]),'position_end',clock(ft[-1]),
          'target_end',clock(tt[-1]),'command_frames_window',sum(events['cid']==0x155),'local_echo',np.count_nonzero(events['flags']&4))
    print('ERRORS',[(clock(x['wall']),hex(int(x['cid'])),x['data'].tobytes().hex()) for x in events[(events['cid']&0x20000000)!=0]])
    nstart=int(datetime.fromisoformat('2026-11-19T15:57:00+08:00').timestamp())*10**9
    native=events[events['cid']==0x155]
    print('NATIVE_FIRST_WINDOW',clock(native['wall'][0]),'last_group_before',clock(tt[np.searchsorted(tt,nstart)-1]))
    print('STATUS',dict(Counter(x.tobytes().hex() for x in events['data'][events['cid']==0x2a1])))
    for j in range(6):
        low=events[events['cid']==0x261+j]
        vals=[struct.unpack('>HhbBH',x.tobytes()) for x in low['data']]
        if vals:
            print('DRIVE',j+1,'status',sorted({hex(v[3]) for v in vals}),'voltage',sorted({v[0]/10 for v in vals}),'last',clock(low['wall'][-1]))
            dip=[(clock(x['wall']),v[0]/10) for x,v in zip(low,vals) if v[0]<220]
            if dip:print('VOLTAGE_LOW',j+1,len(dip),dip[0],dip[-1])
    activities=np.unique(tt[tt>=start]//10**9)
    for sec in activities:
        mask=(tt//10**9==sec); r=(ft//10**9==sec)
        print('ACTIVE',clock(int(sec)*10**9),sum(mask),'command_range_deg',np.ptp(cmd[mask],axis=0)/1000,
              'feedback_range_deg',np.ptp(fb[r],axis=0)/1000 if np.any(r) else None)
    left=np.searchsorted(ft,ft-20_000_000)
    target_right=np.searchsorted(tt,ft,side='right')
    target_left=np.searchsorted(tt,ft-120_000_000,side='right')-1
    candidates=[]
    maximum=[]
    for i in np.flatnonzero((target_left>=0)&(np.arange(len(ft))-left>=2)):
        q=cmd[target_left[i]:target_right[i]]
        movement=fb[i]-fb[left[i]]
        span=np.ptp(q,axis=0)
        for j in np.flatnonzero(abs(movement)>=700):
            sample=(int(abs(movement[j])),int(i),int(j),int(movement[j]),int(span[j]))
            maximum.append(sample)
            if span[j]<=100:candidates.append(sample)
    modes=[('HELD',candidates),('ALL',maximum)]
    qualified=[(i,j,d) for _,i,j,d,_ in candidates if ft[i]>=nstart]
    if qualified:
        i,j,d=min(qualified)
        print('FIRST_SCREEN',clock(ft[left[i]]),clock(ft[i]),'J',j+1,'delta_deg',d/1000)
    for label,items in modes:
        chosen=[]
        for value,i,j,d,span in sorted(items,reverse=True):
            t=ft[i];l=left[i]
            if any(j==axis and abs(t-old)<100_000_000 for old,axis in chosen):continue
            chosen.append((t,j))
            high=events[(events['cid']==0x251+j)&(events['wall']>=ft[l]-10_000_000)&(events['wall']<=t+10_000_000)]
            idx=target_right[i]-1
            print(label,clock(ft[l]),clock(t),'J',j+1,'qpos',fb[l,j]/1000,fb[i,j]/1000,
                  'change',d/1000,'target_span120',span/1000,'last_target',cmd[idx,j]/1000,clock(tt[idx]),
                  'motor',[(clock(x['wall']),*[round(v/1000,3) for v in struct.unpack('>hhi',x['data'].tobytes())[:2]]) for x in high])
            if len(chosen)>=6:break
        if not chosen:print(label,'no candidates')


if __name__=='__main__':audit(Path(sys.argv[1]))
