"""Read-only reconstruction of leader, echoed commands, follower and timing."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
import struct
import sys

import numpy as np


JIDS=(0x155,0x156,0x157)
FIDS=(0x2A5,0x2A6,0x2A7)


def stats(values, unit=1):
    v=np.asarray(values)/unit
    return list(np.round(np.percentile(v,(0,50,95,99,100)),4)) if v.size else []


def show(name, **fields):
    print(name,json.dumps(fields,separators=(',',':'),default=lambda x:x.item()))


def load(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


class Groups:
    def __init__(self, ids, span_ns):
        self.ids,self.span_ns=ids,span_ns
        self.pending=[]
        self.completed=[]
        self.discarded=0

    def push(self,event):
        cid=event['can_id']
        if cid==self.ids[0]:
            self.discarded+=bool(self.pending)
            self.pending=[event];return
        if not self.pending:
            self.discarded+=1;return
        if cid!=self.ids[len(self.pending)] or not 0<=event['mono_ns']-self.pending[0]['mono_ns']<=self.span_ns:
            self.discarded+=1;self.pending=[];return
        self.pending.append(event)
        if len(self.pending)==3:
            q=tuple(v for r in self.pending for v in struct.unpack('>ii',bytes.fromhex(r['data_hex'])))
            self.completed.append((event['mono_ns'],q,self.pending));self.pending=[]


def commands(events):
    accepted=defaultdict(deque)
    assembled=defaultdict(dict)
    reports=Counter()
    result=[]
    for event in events:
        kind,cid=event['kind'],event['can_id']
        if kind=='follower_target_tx_accepted':
            accepted[cid].append(event)
        elif kind=='can0_tx_echo':
            if not accepted[cid] or accepted[cid][0]['data_hex']!=event['data_hex']:
                reports['mismatched_echo']+=1;continue
            source=accepted[cid].popleft()
            reports['confirmed_echo']+=1
            if cid in JIDS:
                key=source.get('source_mono_ns')
                group=assembled[key]
                if cid in group: reports['duplicate_joint_axis_pair']+=1
                group[cid]=event
                if len(group)==3:
                    ordered=[group[i] for i in JIDS]
                    q=[v for r in ordered for v in struct.unpack('>ii',bytes.fromhex(r['data_hex']))]
                    result.append((max(r['mono_ns'] for r in ordered),q,key))
                    del assembled[key]
            elif cid==0x159:
                reports['confirmed_gripper']+=1
    reports['missing_echo']=sum(len(q) for q in accepted.values())
    reports['unassembled_targets']=len(assembled)
    return result,reports


def running(run,session):
    raw=load(run/'can_timeline.jsonl')
    sample=load(run/'samples.jsonl')
    hbs=load(run/'heartbeat.jsonl')
    episode=load(session/'alignment.jsonl')
    origin=episode[0]['sample_monotonic_ns']
    interval=(origin,episode[-1]['sample_monotonic_ns'])
    t_wall_offset=np.median([r['wall']-r['mono'] for r in sample])
    from datetime import datetime
    local=lambda ns: datetime.fromtimestamp(ns/1e9+t_wall_offset).astimezone().isoformat(timespec='milliseconds')
    rel=lambda ns:round((ns-origin)/1e9,5)
    show('RANGE',recording=[local(t) for t in interval],last_control_sample=local(int(sample[-1]['mono']*1e9)),
         last_raw=local(raw[-1]['mono_ns']),result_mtime=session.joinpath('episode_summary.json').stat().st_mtime,
         stop_marker=run.joinpath('STOP').exists(),num_events=len(raw))
    for phase,t0,t1 in [('before',None,interval[0]),('recording',*interval),('after',interval[1],None)]:
        use=lambda s: (t0 is None or s>=t0) and (t1 is None or s<t1)
        use_hb=[r for r in hbs if use(r['mono_ns'])]
        show('TIMING',phase=phase,heartbeat_events=Counter(r['kind'] for r in use_hb),
             heartbeat_ms=stats([r['receive_interval_ms'] for r in use_hb if r.get('receive_interval_ms') is not None]),
             pipe_delay_ms=stats([r['pipe_delay_ms'] for r in use_hb if r.get('pipe_delay_ms') is not None]),
             master_delivery_ms=stats([r['observed_mono_ns']-r['mono_ns'] for r in raw if r['kind']=='master_rx' and r['can_id'] in JIDS and use(r['mono_ns'])],1e6),
             follower_delivery_ms=stats([r['observed_mono_ns']-r['mono_ns'] for r in raw if r['kind']=='follower_rx' and r['can_id'] in FIDS and use(r['mono_ns'])],1e6))
    master=Groups(JIDS,3_000_000); follower=Groups(FIDS,25_000_000)
    for e in raw:
        if e['kind']=='master_rx' and e['can_id'] in JIDS: master.push(e)
        if e['kind']=='follower_rx' and e['can_id'] in FIDS: follower.push(e)
    targets,counts=commands(raw)
    chosen={t:q for t,q,_ in master.completed}
    sourced=[(t,q,s) for t,q,s in targets if s is not None]
    mode=Counter(r['data_hex'] for r in raw if r['kind']=='can0_tx_echo' and r['can_id']==0x151)
    stamps=[r['mono_ns'] for r in raw if r['kind']=='master_rx' and r['can_id'] in JIDS]
    native=[round(r['mono']*1e9) for r in sample]
    show('FORWARDING',counts=counts,mode=mode,native_groups=len(master.completed),follower_groups=len(follower.completed),
         joint_group_frames_discarded=master.discarded,follower_group_frames_discarded=follower.discarded,
         master_time_reversals=sum(b<a for a,b in zip(stamps,stamps[1:])),
         echo_groups=len(targets),compared=len(sourced),
         exact_groups=sum(tuple(q)==chosen.get(s) for _,q,s in sourced),
         source_to_echo_ms=stats([t-s for t,_,s in sourced],1e6),
         joint_group_burst_ms=stats([rows[-1]['mono_ns']-rows[0]['mono_ns'] for _,_,rows in master.completed],1e6),
         source_group_interval_ms=stats(np.diff([t for t,_,_ in sourced]),1e6),
         control_log_ms=stats(np.diff(native),1e6))
    for cid in (0x2a1,*range(0x261,0x267)):
        frames=[x for x in raw if x['kind']=='follower_rx' and x['can_id']==cid]
        if cid==0x2a1:
            states=Counter((e['data_hex'][:6],e['data_hex'][12:]) for e in frames)
            show('STATE',id=hex(cid),flags=[(*a,c) for a,c in states.items()])
        else:
            seen=Counter(int(e['data_hex'][10:12],16) for e in frames)
            extrema=[int(e['data_hex'][:4],16) for e in frames]
            show('DRIVE',joint=cid-0x260,flags=seen,supply_raw=[min(extrema),max(extrema)])
    waves(master.completed,targets,follower.completed,episode,rel,local)


def waves(master,targets,follower,episode,rel,local):
    def series(rows):
        return np.array([r[0] for r in rows],dtype=np.int64),np.asarray([r[1] for r in rows],dtype=float)/1000
    m_t,m_q=series(master);t_t,t_q=series(targets);f_t,f_q=series(follower)
    lookup=np.searchsorted(t_t,f_t,side='right')-1
    t0,t1=episode[0]['sample_monotonic_ns'],episode[-1]['sample_monotonic_ns']
    mask=(lookup>=0)&(f_t>=t0)&(f_t<=t1)
    times, q, goal=f_t[mask],f_q[mask],t_q[lookup[mask]]
    for j in range(6):
        err=q[:,j]-goal[:,j]
        order=np.argsort(np.abs(err))[::-1]
        events=[]
        for i in order:
            if any(abs(times[i]-times[k])<160_000_000 for k in events): continue
            events.append(i)
            if len(events)==3: break
        peaks=[dict(t=rel(times[i]),wall=local(times[i]),target=round(goal[i,j],3),feedback=round(q[i,j],3),error=round(err[i],3)) for i in events]
        show('TRACK',joint=j+1,abs_error_deg=stats(abs(err)),peaks=peaks)
        lead_mask=(m_t[:-1]>=t0)&(m_t[1:]<=t1)
        step=np.diff(m_q[:,j]); order=np.argsort(np.abs(step)*lead_mask)[::-1]
        i=order[0]
        show('LEADER_STEP', joint=j+1,step_deg=round(step[i],3),interval_ms=round((m_t[i+1]-m_t[i])/1e6,3),t=rel(m_t[i+1]),
             target_peak_50ms=round(float(np.max(np.abs(np.diff(goal[:,j],prepend=goal[0,j])))),3))
    checks=[]
    for width in (20_000_000,50_000_000,100_000_000):
        for i,stamp in enumerate(times):
            a=np.searchsorted(times,stamp-width)
            b=np.searchsorted(t_t,stamp-width-100_000_000,side='right')-1
            c=np.searchsorted(t_t,stamp,side='right')
            if a==i or b<0 or c<=b: continue
            condition=np.ptp(t_q[b:c],axis=0)<=.10
            shake=np.abs(q[i]-q[a]);joint=int(np.argmax(np.where(condition,shake,0)))
            if condition[joint] and shake[joint] >= .4:
                checks.append((float(shake[joint]),i,a,joint,width))
    selected=[]
    for delta,i,a,j,width in sorted(checks,reverse=True):
        if any(j==joint and abs(times[i]-t)<150_000_000 for t,joint in selected):continue
        selected.append((times[i],j))
        show('STEADY_TARGET_MOTION',joint=j+1,t=rel(times[i]),wall=local(times[i]),
             window_ms=round((times[i]-times[a])/1e6,3),feedback_change_deg=round(q[i,j]-q[a,j],3),
             feedback_start=q[a,j],feedback_end=q[i,j],target=goal[i,j],window_limit_ms=width/1e6)
        if len(selected)>=12: break
    spikes=[]
    for j in range(6):
        for i,stamp in enumerate(m_t):
            if i<1 or not t0<=stamp<=t1: continue
            step=abs(m_q[i,j]-m_q[i-1,j])
            if step<.6: continue
            idx=np.searchsorted(m_t,stamp+45_000_000)
            if any(abs(v-m_q[i-1,j])<=.2 for v in m_q[i+1:idx,j]): spikes.append((step,j,stamp))
    show('LEADER_ISOLATED_SPIKES',count=len(spikes),largest=[(j+1,rel(t),round(step,3)) for step,j,t in sorted(spikes,reverse=True)[:8]])


if __name__=='__main__':
    running(*map(Path,sys.argv[1:]))
