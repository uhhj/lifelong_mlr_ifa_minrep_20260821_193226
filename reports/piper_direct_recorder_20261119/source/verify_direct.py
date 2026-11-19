"""Verify the desktop capture entry with replay, synthetic images, and the official writer."""
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import io
import json
import socket
import struct
import sys
import time
import numpy as np

import capture
from direct_can import DirectTail, DirectTimeline, IDS, TIMESTAMP, TIMESPEC, WIRE
from verify_capture import verify


def archived(path):
    dtype=np.dtype([('wall','<u8'),('mono','<u8'),('cid','<u4'),('port','u1'),('dlc','u1'),('flags','<u2'),('data','V8')])
    a=np.memmap(path,mode='r',dtype=dtype)
    lo=int(datetime.fromisoformat('2026-11-19T15:57:40+08:00').timestamp())*10**9
    hi=lo+27*10**9
    records=a[(a['wall']>=lo)&(a['wall']<hi)&(a['port']==0)&np.isin(a['cid'],IDS)]
    direct=DirectTimeline()
    native={}
    for row in records:
        cid,ns,raw=int(row['cid']),int(row['wall']),row['data'].tobytes()
        direct.ingest_bus(cid,raw,ns)
        if cid==0x155:
            native={0x155:raw}
        elif cid in (0x156,0x157,0x159):
            native[cid]=raw
    cmd=direct.command_at(ns)
    assert cmd and cmd['origin']=='direct_can_rx' and cmd['frame_hex']==[native[c].hex() for c in (0x155,0x156,0x157,0x159)]
    assert len(direct.targets)==len(records[records['cid']==0x157])==655
    assert direct.observation_at(ns) is not None
    print('ARCHIVE',len(records),'native_groups',len(direct.targets),'held_s',round((ns-cmd['sample_monotonic_ns'])/1e9,4))
    return native


def bounds(native):
    d=DirectTimeline()
    t=10000000
    for cid in (0x155,0x156,0x157):
        d.ingest_bus(cid,native[cid],t);t+=100000
    assert d.command_at(t) is None
    d.ingest_bus(0x159,native[0x159],t)
    assert d.command_at(t) and d.command_at(t)['control_mode_hex'] is None
    future=t+1000000
    d.ingest_bus(0x155,native[0x155],future)
    d.ingest_bus(0x157,native[0x157],future+100000)
    assert d.command_at(t) and d.command_at(future+100000) is None
    for i,c in enumerate((0x155,0x156,0x157)):
        d.ingest_bus(c,native[c],future+5000000+i*100000)
    assert d.command_at(d.watermark)
    prior=d.watermark
    d.ingest_bus(0x2a1,bytes(8),prior+300_000_000)
    assert d.command_at(prior) and d.command_at(d.watermark) is None
    d.ingest_bus(0x159,native[0x159],d.watermark+1)
    assert d.command_at(d.watermark) is None
    for c in (0x155,0x156,0x157):
        d.ingest_bus(c,native[c],d.watermark+1)
    assert d.command_at(d.watermark)
    d.ingest_bus(0x159,b'\x00'*8,d.watermark+1)
    assert d.command_at(d.watermark) is None
    try:d.ingest_bus(0x150,bytes([1])+bytes(7),d.watermark+1)
    except RuntimeError:pass
    else:raise AssertionError('STOP was ignored')


class Camera:
    def __init__(self,level):
        self.start=time.monotonic_ns()
        self.image=np.full((480,640,3),level,dtype=np.uint8)

    def frames(self):
        n=int((time.monotonic_ns()-self.start)//33_333_333)
        return [dict(sequence=i+1,host_monotonic_ns=self.start+i*33_333_333,image=self.image)
                for i in range(max(0,n-59),n+1)]

    def close(self):pass


class ReplaySocket:
    def __init__(self,base,offset,native):
        self.frames=[]
        self.index=0
        for i in range(800):
            t=base+i*5_000_000
            for k,c in enumerate((0x2a5,0x2a6,0x2a7,0x2a8)):
                self.frames.append((t+k*100000,c,struct.pack('>ii',10000,11000)))
            if i==100:
                for k,c in enumerate((0x155,0x156,0x157,0x159)):
                    self.frames.append((t+500000+k*100000,c,native[c]))
        self.offset=offset

    def recvmsg(self,*_):
        if self.index>=len(self.frames) or self.frames[self.index][0]>time.monotonic_ns():
            raise BlockingIOError()
        t,c,raw=self.frames[self.index];self.index+=1
        sec,ns=divmod(t+self.offset,1_000_000_000)
        return WIRE.pack(c,8,raw),[(socket.SOL_SOCKET,TIMESTAMP,TIMESPEC.pack(sec,ns))],0,None

    def close(self):pass


def main(archive,root):
    native=archived(archive)
    bounds(native)
    (root/'official_ui_teleop_v1/binding_runs').mkdir(parents=True)
    (root/'agilex_data_tools/recordings').mkdir(parents=True)
    first=[]
    def open_replay(tail):
        base=time.monotonic_ns()
        first.append(base)
        tail.offset=time.time_ns()-base
        tail.sock=ReplaySocket(base,tail.offset,native)
    old_select=capture.select.select
    stdin=io.StringIO('\n')
    def select_input(read,write,err,timeout=None):
        if read==[stdin]:
            return ([stdin] if first and time.monotonic_ns()-first[0]>2_100_000_000 else []),[],[]
        return old_select(read,write,err,timeout)
    with (patch.object(capture,'check_listen_only'),patch.object(DirectTail,'open',open_replay),
          patch.object(capture,'configured_cameras',return_value={'external':21,'wrist':161}),
          patch.object(capture,'CameraStream',Camera),patch.object(capture.select,'select',select_input),
          patch.object(sys,'stdin',stdin),patch.object(sys,'argv',['capture','--project',str(root)]),
          patch('builtins.input',side_effect=['B1','','a']),patch.object(socket,'socket',side_effect=AssertionError('physical socket forbidden'))):
        capture.main()
    ep,=(root/'agilex_data_tools/recordings').iterdir()
    result=json.loads((ep/'episode_summary.json').read_text())
    assert result['action_source']=='direct_can_rx' and result['source_run'] is None
    assert result['pairs']>=40 and result['valid_feedback']==result['pairs'] and 0<result['valid_targets']<result['pairs']
    verify(ep,result['pairs']-result['valid_targets'],result['pairs'])
    rows=[json.loads(line) for line in (ep/'alignment.jsonl').read_text().splitlines()]
    for row in rows:
        if row['target']:
            assert row['target']['origin']=='direct_can_rx' and row['target']['sample_monotonic_ns']<=row['sample_monotonic_ns']
            assert row['target']['frame_hex']==[native[c].hex() for c in (0x155,0x156,0x157,0x159)]
    print('ENTRY_VERIFIED',result)


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
