"""Summarize archived CAN events crossing the 0.7 degree per-axis hold screen."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import sys
import numpy as np

RECORD=np.dtype([("wall","<u8"),("mono","<u8"),("cid","<u4"),("port","u1"),("size","u1"),("flags","<u2"),("data","V8")])


def inspect(base,name):
    lines=(base/"recordings"/name/"alignment.jsonl").read_text().splitlines()
    start,end=(json.loads(lines[k])["sample_monotonic_ns"] for k in (0,-1))
    chunks=[]
    for p in sorted((base.parent/"diagnostics").glob("piper_passive_*/frames-*.bin")):
        n=p.stat().st_size//RECORD.itemsize
        if not n:
            continue
        raw=np.memmap(p,mode="r",dtype=RECORD,shape=(n,))
        lo,hi=raw["mono"].searchsorted([start-1000000000,end+50000000])
        if hi>lo:
            v=raw[lo:hi]
            chunks.append(v[(v["port"]==0)&(v["size"]==8)].copy())
    raw=np.concatenate(chunks)
    def series(cid,axis):
        data=raw[raw["cid"]==cid]
        pos=np.frombuffer(data["data"].tobytes(),dtype=">i4").reshape(-1,2)[:,axis].astype(float)/1000
        return data["wall"].astype("i8"),pos,data["mono"].astype("i8")
    commands=[series(0x155+j//2,j%2) for j in range(6)]
    events=[]
    for j,(ct,cq,_) in enumerate(commands):
        ft,fq,fm=series(0x2a5+j//2,j%2)
        ch=ct[np.r_[True,np.diff(cq)!=0]]
        earlier=ft.searchsorted(ft-20000000)
        ix=ch.searchsorted(ft,side="right")-1
        hits=(ix>=0)&(ft-ch[ix]>=150000000)&(ft-ft[earlier]>=10000000)&(fm>=start)&(fm<=end)&(abs(fq-fq[earlier])>=.7)
        indexes=np.flatnonzero(hits)
        for group in np.split(indexes,np.flatnonzero(np.diff(ft[indexes])>40000000)+1):
            if not len(group): continue
            i=max(group,key=lambda k:abs(fq[k]-fq[earlier[k]]))
            t=ft[i]; old=earlier[i]
            spans=[]
            for t2,q2,_ in commands:
                l=max(0,t2.searchsorted(t-150000000,side="right")-1)
                r=t2.searchsorted(t,side="right")
                spans.append(round(float(np.ptp(q2[l:r])),3) if r>l else None)
            local=raw[(raw["cid"]==0x251+j)&(raw["wall"]>=ft[old])&(raw["wall"]<=t)]
            drive=np.frombuffer(local["data"].tobytes(),dtype=">i2").reshape(-1,4).astype(float)
            desired=cq[ct.searchsorted(t,side="right")-1]
            events.append(dict(axis=j+1,hits=len(group),at_s=round(float((fm[i]-start)/1e9),3),
                time=datetime.fromtimestamp(int(t)/1e9,tz=timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
                window_ms=round(float(t-ft[old])/1e6,2),jump_deg=round(float(fq[i]-fq[old]),3),
                since_target_change_ms=round(float(t-ch[ix[i]])/1e6,2),target_spans_150ms_deg=spans,
                before_error_deg=round(float(desired-fq[old]),3),after_error_deg=round(float(desired-fq[i]),3),
                peak_abs_drive_current_A=round(float(np.max(abs(drive[:,1])))/1000,3) if len(drive) else None))
    return events


if __name__=="__main__":
    print(json.dumps(inspect(Path(sys.argv[1]),sys.argv[2])))
