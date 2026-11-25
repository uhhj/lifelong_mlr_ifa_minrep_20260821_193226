"""Exercise the real Session dispatch on an archive and a fake actuator."""
from collections import deque
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def verify(code, station, runs):
    sys.path[:0] = [str(code), str(station)]
    from runner import Session, State, PROFILES, SAFETY_NS
    from agilex_arm import AgileXArm
    from bus import DRIVES, TargetGuardFault, bounded_target
    from grip_window import BUNDLE, GripWindow, TIMEOUT_NS
    import runner, bus, agilex_arm, grip_window
    assert all(Path(mod.__file__).resolve().parent == code.resolve()
               for mod in (runner, bus, agilex_arm, grip_window))
    profile = PROFILES["B1-ode50-aligned-20261125"]
    assert profile["sha256"] == BUNDLE
    assert sum(p["sha256"] == BUNDLE for p in PROFILES.values()) == 1
    def records(p):
        return [json.loads(s) for s in p.read_bytes().splitlines()]
    name = "20261125_125325_3006646_B1-ode50-aligned-20261125"
    events = records(runs / (name + ".jsonl"))
    samples = {r["stamp"]: r for r in records(runs/name/"inputs.jsonl") if r["mode_at_query"] == "execute"}
    start = next(r for r in events if r["event"] == "start")
    fault = next(r for r in events if r["event"] == "target_guard_fault")
    sent = [r for r in events if r["event"] == "prediction" and r["mode"] == "execute"]
    received = []
    class FakeArm:
        def command(self, target, receipts):
            received.append(target)
            receipts.extend([0x155,0x156,0x157,0x159])
    state = State(0)
    state.active = True
    rows = []
    session = Session(state, FakeArm(), rows.append, start["mono_ns"], b3_tracking=True, profile=profile)
    session.started = session.last_sent = start["mono_ns"]
    session.previous = start["initial"]
    def feedback(now, q, stamps, torque):
        state.status = (now, bytes([1,0,1,0,0,0,0,0]))
        state.drives = {cid:(now,0x40) for cid in DRIVES}
        state.joints.append((stamps["joint_last_ns"], stamps["joint_first_ns"], q[:6]))
        state.widths.append((stamps["gripper_ns"], q[6], 0x40, torque))
    for r in sent + [fault]:
        now, stamp = r["mono_ns"], r["stamp"]
        frame = samples[stamp]
        # Historical effort is missing. Inject zero, explicitly testing lack of contact evidence.
        feedback(now, r["measured_at_send"], r["measured_at_send_ns"], 0.0)
        session.requested = stamp
        session.beat = now
        session.prediction(dict(stamp=stamp, q=frame["q"], p=frame["p"], full=True, images={},
                                grip_plan_phase=frame["plan"]["alignment"]["phase"]), now)
        if r is not fault:
            assert received[-1] == r["sent"] and session.grip_window.phase == "tracking"
    assert received[-1] == sent[-1]["sent"] and session.grip_window.phase == "waiting"
    assert rows[-1]["gripper_control"]["feedback"]["torque_nm"] == 0
    assert session.last_dispatch_stamp == fault["stamp"]

    def catches(call, expected, match=None):
        try:
            call()
        except expected as exc:
            if match:
                assert match in str(exc), str(exc)
            return
        raise AssertionError("expected a guard fault")

    for delay, message in ((SAFETY_NS, "500 ms"), (TIMEOUT_NS, "超时")):
        now = fault["mono_ns"] + delay
        feedback(now, fault["measured_at_send"],
                 {n:now for n in ("joint_last_ns", "joint_first_ns", "gripper_ns")}, 0.0)
        if delay > SAFETY_NS:
            session.beat = session.predicted = now
        catches(lambda: session.watchdog(now), (ValueError, RuntimeError), message)

    q = [0.0] * 6
    p = [.01] * 6 + [25.]
    def tick(window, t, width, torque, prev, frames, phase="after_close", prediction=p, code=64):
        frames.append((t, width, code, torque))
        req, supervised = window.resolve(t, t, frames, prediction, prev, phase)
        return bounded_target(req, q+[width], prev, .033, b3_tracking=True, supervised_close=supervised)
    def entered():
        obj = GripWindow()
        hist = deque(maxlen=400)
        prev = tick(obj, 1, 61.5, 0, q+[48.75], hist)
        assert obj.phase == "waiting" and prev == q+[48.75]
        return obj, hist, prev
    def confirmed():
        obj, hist, prev = entered()
        for k in range(1,39):
            width = max(38.8, 61.5 - k)
            prev = tick(obj, k*33_000_000, width, -1.02 if width == 38.8 else -.15, prev, hist)
            if obj.phase == "waiting":
                assert prev[:6] == q
            else:
                assert k >= 28 and obj.phase == "holding"
                return obj, hist, prev, k*33_000_000
        raise AssertionError("valid contact did not release arm hold")
    obj, hist, prev, now = confirmed()
    target = tick(obj, now+33_000_000, 38.8, -1, prev, hist)
    assert target[:6] != q and target[6] == prev[6]
    opened = tick(obj, now+66_000_000, 38.8, -1, target, hist, prediction=q+[64.])
    assert opened[6] > target[6] and obj.phase == "tracking"
    obj, hist, prev, now = confirmed()
    catches(lambda: tick(obj, now+33_000_000, 38.8, 0, prev, hist), ValueError, "丢失")
    obj, hist, prev = entered()
    for k in range(1,55):
        prev = tick(obj, k*33_000_000, 61.5, -1.01, prev, hist)
        assert obj.phase == "waiting" and prev == q+[48.75]
    catches(lambda: tick(obj, TIMEOUT_NS+1, 61.5,-1,prev,hist), ValueError, "超时")
    catches(lambda: tick(GripWindow(),1,61.5,0,q+[48.75],deque(),phase="approach"), TargetGuardFault)
    catches(lambda: tick(GripWindow(),1,61.5,0,q+[48.75],deque(),prediction=q+[65]), TargetGuardFault)
    catches(lambda: tick(GripWindow(),1,64,0,q+[48.75],deque()), TargetGuardFault)
    catches(lambda: tick(GripWindow(),1,36,0,q+[50],deque()), TargetGuardFault)
    catches(lambda: tick(GripWindow(),1,61.5,0,q+[48.75],deque(),code=65), ValueError)
    catches(lambda: bounded_target(q+[38], [math.radians(7)]+q[1:]+[61.5], q+[48.75], .03,
                                   b3_tracking=True, supervised_close=True), TargetGuardFault, "J1")
    catches(lambda: bounded_target(q+[38], q+[73], q+[60], .03, b3_tracking=True, supervised_close=True),
            TargetGuardFault, "范围")
    obj, hist, prev = entered()
    catches(lambda: obj.resolve(100_000_000, 1, hist, p, prev, "after_close"), ValueError, "过期")
    catches(lambda: obj.resolve(2, 2, [(2,61.5,64)], p, prev, "after_close"), ValueError, "力矩")
    catches(lambda: obj.resolve(2, 2, [(2,61.5,64,float("nan"))], p, prev, "after_close"), ValueError)
    for widths, torque, spacing in (([39]*12,0,33), ([39]*12,1.0,33), ([39]*12,-1,60),
                                    ([39,40]*6,-1,33), ([39]*12,-1,0)):
        frames = [(100_000_000+i*spacing*1_000_000,w,64,torque) for i,w in enumerate(widths)]
        assert not obj.stable_contact(frames,frames[-1][0])
    inactive = Session(state, FakeArm(), rows.append, 0, profile=PROFILES["B1-ode50-20261123"])
    assert inactive.grip_window is None
    raw = SimpleNamespace(mode=0,grippers_val=39100,status_code=64,grippers_effort=-1043)
    sdk = SimpleNamespace(GetArmGripperMsgs_V3=lambda:SimpleNamespace(time_stamp=123,gripper_state=raw))
    AgileXArm._observe(SimpleNamespace(interface=sdk),state,0x2a8,789,123)
    assert state.widths[-1] == (789,39.1,64,-1.043)
    return dict(profile="B1-ode50-aligned-20261125", archived_commands_matched=len(sent),
                original_fault_replayed_as="waiting", example_mm=received[-1][6],
                archive_torque="missing; zero injected for no-contact case",
                contact_and_release="synthetic feedback", scope="archive and fake actuator only")


if __name__ == "__main__":
    # No sockets are needed by this test, even while importing the installed SDK adapter.
    def offline(event, args):
        if event == "socket.__new__":
            raise RuntimeError("offline verification forbids sockets")
    sys.addaudithook(offline)
    print(json.dumps(verify(*(Path(x) for x in sys.argv[1:])), ensure_ascii=False))
