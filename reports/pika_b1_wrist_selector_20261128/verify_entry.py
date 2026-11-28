"""Exercise both registered weights with virtual controls and archived B1 frames."""
import collections
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

station=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(station))
import runner


def guard_and_selector():
    import desktop
    from guard_case import exercise
    # Drive the real combobox selection, receipt check, preview launch arguments
    # and onsite confirmation handlers without opening a window or worker.
    class Field(dict):
        def set(self,v): self.value=v
        def get(self): return self.value
        def current(self,i): self.set(self['values'][i])
    profiles=runner.PROFILES
    keys=[runner.B1_FIXED_GRIP,runner.B1_WRIST_GRIP]
    labels={}
    cases=('preview_only','return_pending','unknown_mode','confirm_only','outside_pose','usb_unready',
           'success','failed','stop','close','deadline','release','heartbeat','stale_prediction','no_stop_ack')
    for key in keys:
        profile=profiles[key]
        assert runner.uses_fixed_grip(key,profile)
        c=desktop.Console.__new__(desktop.Console)
        c.profiles=profiles;c.task,c.model,c.version=(Field() for i in range(3))
        c.detail=Mock();c.status=Mock();c.controls=Mock();c.rows=Mock();c.confirm=Mock();c.command=Mock()
        c.task.set('B1 '+profile['label']);c.choose_task();c.model.set('ODE');c.choose_model()
        assert len(c.choices)==len([v for v in profiles.values() if v['family']=='ODE' and v['task']=='B1'])
        c.version.set(profile['version']);c.selected()
        assert c.key==key and c.verified and profile['sha256'][:12] in c.detail.set.call_args.args[0]
        c.proc=c.manual=None
        with patch.object(desktop.subprocess,'Popen') as popen:
            c.preview()
            argv=popen.call_args.args[0]
            assert argv[-3]=='guard' and argv[-1]==key and Path(argv[2])==station/'runner.py'
            c.link.close()
        c.ready=True;c.begin()
        assert c.mode=='confirm' and c.command.call_args.args==('start_clicked',)
        c.accept()
        assert c.mode=='pending' and c.command.call_args.args==('confirm',)
        labels[key]=dict(label=profile['version'],sha256=profile['sha256'],cases=[exercise(case,key) for case in cases])
    for key in keys:
        swapped=profiles[keys[1-keys.index(key)]]
        try: runner.uses_fixed_grip(key,swapped)
        except ValueError: pass
        else: raise AssertionError('mismatched weight accepted')
    return labels


def inference():
    import cv2
    import numpy as np
    import torch
    from bus import bounded_target
    from b1_placement import B1Placement
    cv2.setNumThreads(1);torch.set_num_threads(4)
    folder=station.parent/'policy_runs'/'20261128_135114_1209855_B1-ode50-aligned-20261125'
    def lines(p):
        return [json.loads(line) for line in p.read_bytes().splitlines()]
    records=lines(folder.with_suffix('.jsonl'))
    start=next(r for r in records if r['event']=='gripper_stable')
    sent=[r for r in records if r.get('mode')=='execute' and r.get('sent')]
    inputs={r['stamp']:r for r in lines(folder/'inputs.jsonl')}
    sequence=[(inputs[t],None) for t in inputs[sent[0]['stamp']]['history'][:-1]]+[(inputs[r['stamp']],r) for r in sent]
    output={}
    class Finished(Exception): pass
    for key in (runner.B1_FIXED_GRIP,runner.B1_WRIST_GRIP):
        model=runner.load_policy(runner.PROFILES[key])
        expected='ode_wrist_alignment' if key==runner.B1_WRIST_GRIP else 'ode_alignment'
        assert type(model).__module__==expected
        gate=B1Placement();counts=collections.Counter();times=[];goals=[]
        prev=start['initial'];ack=0;last_send=start['mono_ns'];i=0
        state=SimpleNamespace(now=0)
        def pair(cameras,prior):
            nonlocal i
            if i==len(sequence):raise Finished
            e,row=sequence[i];im=cv2.imread(str(folder/e['images']['model']));assert im.shape==(480,1280,3)
            state.entry=e;state.row=row;state.now=e['stamp'];i+=1
            return tuple(dict(e['cameras'][role],image=view) for role,view in
                         zip(('external','wrist'),(im[:,:640],im[:,640:])))
        def receive(link):
            active=state.row is not None
            e=state.entry
            return dict(q=e['q'],feedback_ns=e['feedback_ns'],mode_at_query='execute' if active else 'preview',
                        execution=start['mono_ns'] if active else 0,previous_stamp=ack,previous_sent=prev if active else None,
                        gripper_phase=gate.phase)
        def emit(link,msg):
            nonlocal prev,ack,last_send
            if msg['op']!='predict' or state.row is None:return
            row=state.row;t=msg['stamp'];measured=row['measured_at_send'];now=row['send_started_ns']
            assert msg['full'] and msg['gate_execution']==start['mono_ns'] and msg['grasp_alignment']['stamp']==t
            assert msg['grasp_alignment']['aligned']==row['grasp_alignment']['aligned']
            assert msg['close_intent']==row['close_intent']
            arm=bounded_target(msg['p'],measured,prev,(now-last_send)/1e9,b3_tracking=True,arm_only=True)
            goal=gate.next_goal(t,msg['close_intent'],msg['tip_xyz_mm'],measured[6],vision=msg['grasp_alignment'])
            if goal is not None:goals.append(goal)
            prev=[*arm,prev[6] if goal is None else goal];ack=t;last_send=now
            counts[model.trace['offset']]+=1
        class Archive:
            def predict(self,model,pair,pixels,query,stamp):
                begin=time.perf_counter_ns()
                result=model.predict_recorded(pixels,query,stamp)
                if state.row is not None:times.append((time.perf_counter_ns()-begin)/1e6)
                return result,state.entry['images']
        clock=SimpleNamespace(monotonic=time.monotonic,monotonic_ns=lambda:state.now)
        try:
            with patch.multiple(runner,time=clock,next_pair=pair,send=emit,receive=receive):
                runner.infer_loop(None,None,model,Archive(),runner.uses_fixed_grip(key,runner.PROFILES[key]))
        except Finished: pass
        assert len(times)==len(sent) and counts[7]>0 and not goals
        output[key]=dict(frames=len(times),backend=expected,eighth_steps=counts[7],gripper_events=goals,
                         device_writes=0,p95_model_ms=round(float(np.percentile(times,95)),2),max_model_ms=round(max(times),2))
        del model;gc.collect();torch.cuda.empty_cache()
    return output

if __name__=='__main__':
    print(json.dumps(guard_and_selector() if sys.argv[2]=='guard' else inference(),ensure_ascii=True))
