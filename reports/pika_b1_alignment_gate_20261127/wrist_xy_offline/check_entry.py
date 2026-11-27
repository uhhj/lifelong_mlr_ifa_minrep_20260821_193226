"""Exercise the installed guard protocol with a virtual clock, camera, arm and gripper.

All device constructors, child processes, CAN binding and sockets are replaced
before entering guard(). The guard uses a temporary directory. Only the test
summary is retained beside this script.
"""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
from unittest.mock import patch
import sys

import runner
from bus import DRIVES
from b1_placement import PLACE_XYZ_MM


class Endpoint:
    def __init__(self):
        self.incoming = {}
        self.outgoing = []

    def recv(self, size): return json.dumps(self.incoming).encode()
    def sendall(self, buf): self.outgoing.append(json.loads(buf))
    def fileno(self): return -1
    def close(self): pass
    def setblocking(self, flag): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


def exercise(case):
    key = runner.B1_FIXED_GRIP
    profile = dict(runner.PROFILES[key])
    assert profile['end_mode'] == 'stop'
    if case == 'return_pending':
        profile.pop('end_mode')
    if case == 'unknown_mode':
        profile['end_mode'] = 'unrecognised'
    angles = [math.radians((lo+hi)/2) for lo,hi in profile['start_range'][:6]]
    if case == 'outside_pose':
        angles[0] += math.radians(30)
    clock = SimpleNamespace(now=10_000_000_000)
    clock.monotonic_ns = lambda: clock.now
    clock.time_ns, clock.strftime = time.time_ns, time.strftime
    clock.sleep = lambda seconds: setattr(clock, 'now', clock.now+int(seconds*1e9))
    ui, inference, child = Endpoint(), Endpoint(), Endpoint()
    events = []
    step = 0
    stamp = 0
    attempts = 0
    mode_started = 0
    stop_requested = False

    class Arm:
        version = 'offline'
        receiver = object()
        def __init__(self, port): pass
        def close(self): pass
        def drain(self, state):
            stopped = stop_requested and case != 'no_stop_ack'
            status = bytes([1,int(stopped),1,0,0,0,0,0])
            state.status = (clock.now,status)
            state.joints.append((clock.now,clock.now,list(angles)))
            state.drives.update({cid:(clock.now,0x40) for cid in DRIVES})
        def start_joint_mode(self):
            nonlocal mode_started
            mode_started = step
            events.append('joint_mode')
        def command(self, target, ids): events.append('arm_target')
        def stop(self):
            nonlocal stop_requested
            stop_requested = True
            events.append('CAN_STOP')

    class Grip:
        armed = False
        bounds = (0,80)
        width = 63.
        device = SimpleNamespace(motor_window=lambda t:[])
        def close(self): pass
        def drain(self, state):
            state.widths.append((clock.now,self.width,0x40 if self.armed else 0,0))
        def check_start(self):
            if case == 'usb_unready':
                raise ValueError('模拟夹爪反馈不稳定')
        def switch(self, enable):
            if enable:
                self.check_start()
            events.append('USB_ENABLE' if enable else 'USB_DISABLE')
            self.armed = enable
            self.ready_after_ns = clock.now + 700_000_000
        def verify_feedback(self): assert self.armed
        def command_once(self, goal):
            self.width = goal
            events.append('width_' + str(int(goal)))

    def select(ports, *args):
        nonlocal step, stamp, attempts
        step += 1
        if step > 580: raise AssertionError('virtual session exceeded its time budget')
        # Query then return a prediction for the exact same double-camera stamp.
        if step % 2:
            clock.now += 32_333_333
            stamp = clock.now
            inference.incoming = {'op':'query','stamp':stamp}
            op = 'beat'
            if step == 83 and case != 'confirm_only': op = 'start_clicked'
            if step == 85: op = 'confirm'
            if step==83 and case=='preview_only': op='close'
            if case in ('return_pending','unknown_mode','usb_unready','outside_pose','confirm_only') and step==97:
                op='close'
            if case in ('success','failed','stop','close','no_stop_ack') and mode_started and step >= mode_started+8:
                op = 'success' if case=='no_stop_ack' else case
            if case=='heartbeat' and mode_started and op=='beat':
                return [inference], [], []
            ui.incoming={'op':op}
            return [ui,inference],[],[]
        clock.now += 1_000_000
        q = inference.outgoing[-1]
        phase = q.get('gripper_phase')
        visual = case=='release'
        tip = list(PLACE_XYZ_MM)
        if phase=='approach': tip[2]-=22
        inference.incoming=dict(op='predict',stamp=stamp,p=q['q'],q=q['q'],full=case!='stale_prediction' or not mode_started,
                                gate_execution=q['execution'],tip_xyz_mm=tip,grasp_intent_text='模拟预测',
                                close_intent=visual,grasp_alignment={'stamp':stamp,'aligned':visual})
        if not q['q']: raise AssertionError('missing synthetic feedback')
        return [inference],[],[]

    def binding(*,allow_listen_only):
        nonlocal attempts
        if not allow_listen_only: attempts+=1

    class Worker:
        def __init__(self,*args,**kwargs): pass
        def poll(self): return 0

    with tempfile.TemporaryDirectory() as temp:
        w=Path(temp)
        measured=json.loads(Path(__file__).with_name('outcomes.json').read_bytes())
        assert measured['sha256']==profile['sha256']
        positives=[r for r in measured['controls']['episodes']['other'] if r['close_accepted']]
        assert len(positives)>=3
        evidence=dict(profile=key,sha256=profile['sha256'],windows=positives)
        (w/f'verified-{key}.json').write_text(json.dumps(evidence))
        with patch.multiple(runner, HERE=w,WORK=w,SYNC=w,time=clock,AgileXArm=Arm,
                            socket=SimpleNamespace(SOCK_SEQPACKET=0,socketpair=lambda **kw:(inference,child)),
                            select=SimpleNamespace(select=select),subprocess=SimpleNamespace(Popen=Worker)), \
             patch.dict(sys.modules,stop_link=SimpleNamespace(check_path=binding),b1_gripper=SimpleNamespace(B1Gripper=Grip)), \
             patch.dict(runner.PROFILES,{key:profile}):
            runner.guard(ui,key,None)
        lines=[json.loads(line) for line in next((w/'policy_runs').glob('*.jsonl')).read_text().splitlines()]
    end=lines[-1]
    gate_locked=case in ('return_pending','unknown_mode')
    ready = [msg for msg in ui.outgoing if msg.get('ready')]
    refusals=[r.get('detail') for r in lines if r.get('event')=='start_rejected']
    assert lines[0]['end_mode']==('return_pending' if gate_locked else 'stop')
    assert ready or gate_locked or case in ('outside_pose','usb_unready')
    if case in ('preview_only','return_pending','unknown_mode','outside_pose','usb_unready','confirm_only'):
        assert events==[] and end['outcome']=='preview'
        if gate_locked:
            assert not ready and runner.B1_RETURN_PENDING in refusals
        if case=='outside_pose':
            assert any('起始位姿' in err for err in refusals)
        if case=='usb_unready':
            assert any('夹爪反馈不稳定' in err for err in refusals)
        if case=='confirm_only':
            assert not attempts and any('现场窗口确认' in err for err in refusals)
        return dict(case=case,ready=bool(ready),outcome=end['outcome'],motor_calls=0)
    assert attempts==1 and any(r.get('event')=='start_confirmed' for r in lines)
    assert events[:2]==['USB_ENABLE','joint_mode'] and events[-2:]==['CAN_STOP','USB_DISABLE'],(case,events,end)
    assert events.count('CAN_STOP')==1 and not any(r['event']=='b1_end_phase' or 'end_phase' in r for r in lines)
    expected=case if case in ('success','failed') else 'aborted'
    assert end['outcome']==expected,(case,end)
    ack='已收到从臂 STOP 反馈' in end['detail']
    assert ack == (case!='no_stop_ack')
    assert 'Pika 已反馈失能' in end['detail']
    reason={'deadline':'6.3 秒结束','release':'放置松开反馈','heartbeat':'500 ms','stale_prediction':'八帧历史中断','no_stop_ack':'STOP 未确认'}
    if case in reason: assert reason[case] in end['detail'],(case,end)
    if case=='release': assert 'width_35' in events and 'width_60' in events
    return dict(case=case,ready=bool(ready),outcome=end['outcome'],stop_count=events.count('CAN_STOP'),
                usb_disable_count=events.count('USB_DISABLE'),stop_ack=ack)


if __name__=='__main__':
    cases=('preview_only','return_pending','unknown_mode','confirm_only','outside_pose','usb_unready',
           'success','failed','stop','close','deadline','release','heartbeat','stale_prediction','no_stop_ack')
    results=[exercise(case) for case in cases]
    Path(__file__).with_name('guard_cases.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results))
