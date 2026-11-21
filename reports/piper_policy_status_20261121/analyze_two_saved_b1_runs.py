"""Reconcile an ended execution and image index with its independently archived CAN slice."""
from bisect import bisect_right, bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
from statistics import median
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/piper_executor"))
from bus import LIMITS, bounded_target, target_frames


def read(path):
    return [json.loads(l) for l in path.read_bytes().splitlines()]


def units(vec):
    return [round(math.degrees(v) if i<6 else v,4) for i,v in enumerate(vec)]


def clock(w):
    return datetime.fromtimestamp(w/1e9,timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def analyze(folder):
    main=read(folder/"main.jsonl")
    archive={r["stamp"]:r for r in read(folder/"inputs.jsonl")}
    start,= [r for r in main if r["event"]=="start"]
    end=main[-1]
    assert end["event"]=="end"
    execution=[r for r in main if r.get("mode")=="execute" and r["event"]=="prediction"]
    assert all(all(r[k]==archive[r["stamp"]][k] for k in ("q","p","images","feedback_ns")) for r in execution)
    records=sorted((w,c,b,f) for w,_,c,port,n,f,b in struct.iter_unpack("<QQIBBH8s",(folder/"can.bin").read_bytes()) if not port and n==8)
    by=defaultdict(list)
    for w,c,b,f in records:by[c].append((w,b,f))
    times={c:[w for w,b,f in v] for c,v in by.items()}
    # Recover the guard's realtime-minus-monotonic offset using the same gripper RX kernel timestamps.
    approx=main[0]["wall_ns"]-main[0]["mono_ns"]
    offsets=[]
    for r in execution:
        t=r["feedback_ns"]["gripper_ns"]
        i=bisect_left(times[0x2a8],t+approx)
        w,b,_=min(by[0x2a8][max(i-1,0):i+1],key=lambda v:abs(v[0]-t-approx))
        assert abs(t+approx-w)<2_000_000 and abs(struct.unpack_from(">i",b)[0]/1000-r["q"][6])<1e-5
        offsets.append(w-t)
    anchor=int(median(offsets))
    assert max(abs(v-anchor) for v in offsets)<100_000
    begin=start["mono_ns"]+anchor
    finish=end["mono_ns"]+anchor
    stops=[w for w,b,f in by[0x150] if begin<=w<=finish+10_000_000 and b[0]==1 and f&4]
    stop=stops[0]
    ack=next(w for w,b,f in by[0x2a1] if w>=stop and b[1]==1 and not f&4)
    def latest(c,t):
        i=bisect_right(times[c],t)-1
        if i<0:raise ValueError("Missing earlier CAN feedback for "+hex(c))
        return by[c][i][1]
    def feedback(t):
        return [math.radians(i/1000) for c in (0x2a5,0x2a6,0x2a7) for i in struct.unpack(">ii",latest(c,t))]+[struct.unpack_from(">i",latest(0x2a8,t))[0]/1000]
    def pose(t):
        return [round(struct.unpack_from(">i",latest(c,t),i)[0]/1000,2) for c,i in ((0x2a2,0),(0x2a2,4),(0x2a3,0))]
    expected=defaultdict(list)
    for r in execution:
        for c,b in target_frames(r["sent"]):expected[c].append(b)
    for c,items in expected.items():
        assert items==[b for w,b,f in by[c] if begin<=w<stop and f&4],hex(c)
    inputs=sorted((r for r in archive.values() if begin<=r["stamp"]+anchor<=stop),key=lambda r:r["stamp"])
    submitted_stamps={s["stamp"] for s in execution}
    unmatched=[r for r in inputs if r["stamp"] not in submitted_stamps]
    last=execution[-1]
    prev=last["sent"]
    # The guard checks fresh feedback against its previous submitted target (not its new prediction).
    # Sample a conservative interval bracketing the unlogged guard read using its independent feedback frames.
    vals=[]
    if unmatched:
        lower=max(last["mono_ns"]+anchor,unmatched[-1]["predicted_ns"]+anchor-6_000_000)
        times_test=sorted({lower,stop-1,*[w for w,c,b,f in records if lower<w<stop and c in (0x2a5,0x2a6,0x2a7,0x2a8)]})
        for t in times_test:
            v=feedback(t)
            residual=units([a-b for a,b in zip(prev,v)])
            axis=[i+1 for i,val in enumerate(residual) if abs(val)>(5 if i<6 else 12)]
            outside=[i+1 for i,(x,(lo,hi)) in enumerate(zip(units(v),LIMITS+[(0,70)]))
                     if not lo-(1 if i<6 else 2)<=x<=hi+(1 if i<6 else 2)]
            try:
                bounded_target(unmatched[-1]["p"],v,prev,1/30)
                rejected=False
            except ValueError:rejected=True
            vals.append(dict(t=t,feedback=units(v),residual=residual,axes=axis,outside=outside,rejected=rejected))
    errors=[dict(at=clock(w),ms_before_stop=round((stop-w)/1e6,3),can_id=hex(c),data=b.hex())
            for w,c,b,f in records if begin<=w<=ack and c&0x20000000]
    rates=[]
    for age in (0,1,2,3,4,5,5.5,(last["mono_ns"]-start["mono_ns"])/1e9):
        t=begin+int(age*1e9)
        if t>last["mono_ns"]+anchor+1000:continue
        r=min(execution,key=lambda x:abs(x["mono_ns"]+anchor-t))
        rates.append(dict(sec=round((r["mono_ns"]-start["mono_ns"])/1e9,3),
                          j5_pred=units(r["p"])[4],j5_sent=units(r["sent"])[4],j5_meas=units(r["measured_at_send"])[4],
                          j1_sent=units(r["sent"])[0],grip_meas=r["measured_at_send"][6],xyz_mm=pose(r["mono_ns"]+anchor)))
    post_stop=[]
    for ms in (0,50,100,250,500,1000):
        t=stop+ms*1_000_000
        for cid in (0x2a7,0x255,0x265,0x2a1):
            position=bisect_right(times[cid],t)-1
            assert position >= 0 and t-times[cid][position]<50_000_000, "Stale post-STOP feedback"
        speed,current,motor_pos=struct.unpack(">hhi",latest(0x255,t))
        post_stop.append(dict(ms=ms,j5_deg=units(feedback(t))[4],motor_speed_raw=speed,motor_pos_raw=motor_pos,
                              motor_current_raw=current,drive_byte=latest(0x265,t)[5],arm_status=latest(0x2a1,t)[1]))
    post_commands=Counter(hex(c) for w,c,b,f in records if stop<w<=stop+10**9 and
                          (0x150<=c<=0x15f or c in (0x470,0x471)))
    normal_drive={str(c-0x260):sorted({b[5] for w,b,f in by[c] if begin<=w<=stop}) for c in range(0x261,0x267)}
    mode_commands=[b.hex() for w,b,f in by[0x151] if begin-1_000_000<=w<=stop and f&4]
    return dict(begin=clock(begin),end=clock(finish),duration_sec=round((finish-begin)/1e9,6),outcome=end["outcome"],detail=end["detail"],
                committed_targets=len(execution),matching_wire_counts={hex(c):len(v) for c,v in expected.items()},
                external_targets=sum(1 for w,c,b,f in records if begin<=w<stop and c in expected and not f&4),
                decision_inputs=len(inputs),unsubmitted=len(unmatched),history_complete=all(len(set(r["history"]))==8 and all(h in archive for h in r["history"]) for r in inputs),
                max_view_skew_ms=round(max(r["skew_ns"] for r in inputs)/1e6,3),
                max_input_gap_ms=round(max((b["stamp"]-a["stamp"])/1e6 for a,b in zip(inputs,inputs[1:])),3),
                max_feedback_age_ms=round(max((r["stamp"]-min(r["feedback_ns"].values()))/1e6 for r in inputs),3),
                drive_status=normal_drive,mode_commands=mode_commands,
                aligned_offset_spread_ns=max(offsets)-min(offsets),
                errors=errors,stop=clock(stop),stop_ack_ms=round((ack-stop)/1e6,3),
                start_feedback=units(start["initial"]),final_sent=units(prev),final_prediction=units(unmatched[-1]["p"]) if unmatched else None,
                feedback_at_stop=units(feedback(stop-1)),feedback_after_stop_1s=units(feedback(stop+10**9)),
                max_command_delta_from_start_deg=round(max(abs(math.degrees(r["sent"][i]-start["initial"][i])) for r in execution for i in range(6)),3),
                max_grip_delta_from_start_mm=round(max(abs(r["sent"][6]-start["initial"][6]) for r in execution),3),
                motion=rates,post_stop=post_stop,post_stop_control_ids_1s=post_commands,
                guard_window=dict(samples=len(vals),start=vals[0] if vals else None,end=vals[-1] if vals else None,
                    axes_over=sorted({tuple(v["axes"]) for v in vals}),outside=sorted({tuple(v["outside"]) for v in vals}),
                    all_rejected=bool(vals) and all(v["rejected"] for v in vals),
                    j5_tracking_deg_range=[min(v["residual"][4] for v in vals),max(v["residual"][4] for v in vals)] if vals else None))


if __name__=="__main__":
    print(json.dumps([analyze(Path(s)) for s in sys.argv[1:]],indent=2,ensure_ascii=False,allow_nan=False))
