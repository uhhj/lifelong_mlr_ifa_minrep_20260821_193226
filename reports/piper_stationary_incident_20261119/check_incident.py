"""Summarize a frozen, single-port PiPER trace near the last received feedback."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta
import numpy as np

LAYOUT=np.dtype([('wall','<u8'),('mono','<u8'),('cid','<u4'),('port','u1'),
                 ('dlc','u1'),('flags','<u2'),('data','V8')])


def clock(ns):
    seconds,nanos=divmod(int(ns),10**9)
    return datetime.fromtimestamp(seconds,timezone(timedelta(hours=8))).strftime('%H:%M:%S')+f'.{nanos//1000:06d}'


def check(path):
    meta=json.loads(path.with_name('manifest.json').read_text())
    with path.open('rb') as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    if path.stat().st_size!=meta['bytes'] or digest.hexdigest()!=meta['sha256']:
        raise ValueError('Frozen file differs from its manifest')
    raw=np.memmap(path,mode='r',dtype=LAYOUT)
    print('coverage',clock(raw['wall'][0]),clock(raw['wall'][-1]),'frames',len(raw))
    print('default_control',np.count_nonzero(np.isin(raw['cid'],[0x150,0x151,0x155,0x156,0x157,0x159,0x470,0x471])),
          'local_echo',np.count_nonzero(raw['flags']&4),'can_errors',np.count_nonzero(raw['cid']&0x20000000))
    tail=raw[raw['wall']>=int(raw['wall'][-1])-120*10**9]
    print('status',[(b.tobytes().hex(),int(n)) for b,n in zip(*np.unique(tail['data'][tail['cid']==0x2a1],return_counts=True))])
    for j in range(6):
        pos=tail[(tail['cid']==0x2a5+j//2)&(tail['dlc']==8)]
        t=pos['wall'].astype('int64')
        angles=np.frombuffer(pos['data'].tobytes(),dtype='>i4').reshape(-1,2)[:,j%2].astype('i8')
        left=np.searchsorted(t,t-20_000_000)
        delta=angles-angles[left]
        ok=(t-t[left]>=9_000_000)&(abs(delta)>=700)
        hi=tail[(tail['cid']==0x251+j)&(tail['dlc']==8)]
        motors=np.frombuffer(hi['data'].tobytes(),dtype='>i2').reshape(-1,4)
        if not np.any(ok):continue
        first=np.flatnonzero(ok)[0]
        peak=np.argmax(np.where(ok,abs(delta),0))
        k=np.searchsorted(hi['wall'],t[peak],side='right')-1
        print('joint',j+1,'first_0.7deg_window',clock(t[left[first]]),clock(t[first]),
              'peak_window',clock(t[left[peak]]),clock(t[peak]),'delta_degrees',delta[peak]/1000,
              'motor_sample',clock(hi['wall'][k]),'reported_radps_A',list(motors[k,:2].astype(float)/1000))


if __name__=='__main__':check(Path(sys.argv[1]))
