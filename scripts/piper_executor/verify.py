"""Offline verification with recorded images and isolated Unix socket pairs."""
import json
import os
from pathlib import Path
import select
import socket
import struct
import sys
import tempfile
import threading
import time
from unittest.mock import patch


def forbid_hardware(event,args):
    if event=="socket.__new__" and args[1]==socket.AF_CAN:
        raise AssertionError("Offline verification forbids CAN")
    if event=="open" and isinstance(args[0],str) and args[0].startswith("/dev/video"):
        raise AssertionError("Offline verification forbids cameras")


sys.addaudithook(forbid_hardware)
import bus
import runner


def simulated_session(case):
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td)
        a,b=socket.socketpair(type=socket.SOCK_SEQPACKET)
        rx,tx=socket.socketpair(type=socket.SOCK_DGRAM)
        rx.setblocking(False)
        rx.setsockopt(socket.SOL_SOCKET,bus.TS,1)
        sent=[]
        stopped=threading.Event()
        close=threading.Event()
        def frame(cid,data):
            tx.send(bus.WIRE.pack(cid,8,data))
        class SimBus(bus.Bus):
            def __init__(self,port):
                self.sock=rx
                self.offset=time.time_ns()-time.monotonic_ns()
            def send(self,cid,data):
                sent.append((cid,data))
                if cid==0x150: stopped.set()
        def plant():
            while not close.is_set():
                frame(0x2A1,bytes((1,int(stopped.is_set()),1,0,0,0,0,0)))
                for i in bus.DRIVES: frame(i,bytes.fromhex("0000000000400000"))
                for i in bus.JOINT: frame(i,bytes(8))
                frame(0x2A8,bytes.fromhex("00004e2000004000"))
                if case=="leader": frame(0x155,bytes(8))
                if case=="conflict" and any(i==0x159 for i,_ in sent): frame(0x156,bytes(8))
                close.wait(.005)
        def mock_worker(link):
            try:
                while not close.wait(.025):
                    stamp=time.monotonic_ns()
                    runner.send(link,dict(op="query",stamp=stamp))
                    q=runner.receive(link)["q"]
                    if q is not None:
                        runner.send(link,dict(op="predict",stamp=stamp,q=q,p=[.002,.002,-.002,0,0,0,21],full=True))
            except (EOFError,OSError): pass
            finally: link.close()
        class Worker:
            def __init__(self,args,**kwargs):
                self.link=socket.socket(fileno=os.dup(kwargs["pass_fds"][0]))
                self.thread=threading.Thread(target=mock_worker,args=(self.link,))
                self.thread.start()
            def poll(self): return None if self.thread.is_alive() else 0
            def terminate(self):
                close.set()
                try: self.link.shutdown(socket.SHUT_RDWR)
                except OSError: pass
            def wait(self,t=None): self.thread.join(t)
        from types import SimpleNamespace
        errors=[]
        def guarded():
            try:
                with b: runner.guard(b,0,"test")
            except Exception as e: errors.append(repr(e))
        prod=threading.Thread(target=plant)
        guard=threading.Thread(target=guarded)
        with patch.object(runner,"WORK",folder),patch.object(runner,"SYNC",folder),patch.object(runner,"Bus",SimBus),patch.object(runner.subprocess,"Popen",Worker),patch.dict(sys.modules,stop_link=SimpleNamespace(check_path=lambda **kw:None)):
            prod.start()
            guard.start()
            started=time.monotonic()
            denied=asked=active=finish=False
            try:
                while time.monotonic()-started<5:
                    if not(active and case=="heartbeat"):
                        try: runner.send(a,dict(op="beat"))
                        except BrokenPipeError: pass  # The final status may already be queued.
                    if select.select([a],[],[],.015)[0]:
                        try: msg=runner.receive(a)
                        except EOFError: raise AssertionError((case,errors,sent[-2:]))
                        if "pred" in msg and not asked and time.monotonic()-started>1.15:
                            assert not sent
                            runner.send(a,dict(op="close" if case=="preview" else "start"))
                            asked=True
                        if "error" in msg:
                            denied=True
                            runner.send(a,dict(op="close"))
                        active=active or msg.get("active",False)
                        if case=="success" and active and not finish and any(i==0x159 for i,_ in sent):
                            runner.send(a,dict(op="success"))
                            finish=True
                        if msg.get("done"):break
                    time.sleep(.006)
                else: raise AssertionError("guard timeout "+case)
            finally:
                close.set()
                a.close()
                prod.join(1)
                guard.join(2)
                tx.close()
        assert not errors and not guard.is_alive(),errors
        rows=[json.loads(s) for s in next((folder/"policy_runs").glob("*.jsonl")).read_text().splitlines()]
        last=rows[-1]
        print(case,last,flush=True)
        if case in ("preview","leader"):
            assert not sent and last["outcome"]=="preview" and denied==(case=="leader")
        else:
            assert sent[0][0]==0x151 and sent[-1][0]==0x150 and last["outcome"]==("success" if case=="success" else "aborted"),last
            assert all(i in {0x151,0x155,0x156,0x157,0x159,0x150} for i,_ in sent)
            assert any(r.get("mode")=="execute" for r in rows)
            assert "已收到从臂 STOP" in last["detail"],last
            if case=="heartbeat": assert "500 ms" in last["detail"],last
            if case=="conflict": assert "其他" in last["detail"],last
        return dict(case=case,outcome=last["outcome"],stop_confirmed=stopped.is_set(),simulated_frames=len(sent))


def verify(work):
    import cv2
    import numpy as np
    import torch
    from policy import Predictor,BEST
    torch.set_num_threads(4)
    tests=[simulated_session(c) for c in ("preview","leader","success","conflict","heartbeat")]
    q=[0,0,0,0,0,0,20]
    assert dict(bus.target_frames(q))[0x159]==bytes.fromhex("00004e2003e80100")
    state=bus.State(0)
    for i,t in zip(bus.JOINT,(1,4_000_002,4_000_003)): state.ingest(i,bytes(8),t)
    assert not state.joints
    samples=[]
    model=Predictor(work,0)
    from piper_base import read_episodes
    episodes,_=read_episodes(work)
    for task in range(3):
        model.task=task
        model.tag.fill_(task)
        model.history.clear()
        ep=next(e for e in episodes if e["task"]==task and e["split"]=="val")
        mid=len(ep["q"])//2
        ids=list(range(mid-7,mid+1))
        for i in ids:
            t=time.monotonic()
            result=model.predict([cv2.imread(str(view[i])) for view in ep["cameras"]],ep["q"][i])
            elapsed=(time.monotonic()-t)*1000
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            v=model.front.encode(ep["cameras"],ids,task)[None].cuda()
            _,expected=model.model(v,torch.from_numpy(ep["q"][ids])[None].cuda(),model.tag)
        diff=np.abs(np.array(result)-expected[0].float().cpu().numpy())
        assert np.max(np.rad2deg(diff[:6]))<.1 and diff[6]<.5,(task,diff)
        samples.append(dict(task="B"+str(task+1),max_delta_deg=float(np.max(np.rad2deg(diff[:6]))),
                            delta_mm=float(diff[6]),last_pair_ms=round(elapsed,2)))
    return dict(checkpoint=BEST,simulated=tests,replay=samples)


if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve()),allow_nan=False))
