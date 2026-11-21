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
from collections import deque
from unittest.mock import patch


def forbid_hardware(event,args):
    if event=="socket.__new__" and args[1]==socket.AF_CAN:
        raise AssertionError("Offline verification forbids CAN")
    if event=="open" and isinstance(args[0],str) and args[0].startswith("/dev/video"):
        raise AssertionError("Offline verification forbids cameras")


sys.addaudithook(forbid_hardware)
import bus
import runner


def synthetic_view(role,seq):
    import numpy as np
    image=np.empty((480,640,3),np.uint8)
    image[:]=(21+90*role,seq%175+30,220-80*role)
    image[:240,:320,0]=240
    return image


def camera_worker(link,directory,case):
    from capture import FrameArchive
    class Camera:
        def __init__(self,role):
            self.role=role
        def frames(self):
            # A shared 30 Hz grid, with the wrist arriving 1 ms after the overhead view.
            tick=33_333_333
            seq=(time.monotonic_ns()-1_000_000)//tick
            stamp=seq*tick+self.role*1_000_000
            return [dict(sequence=seq,host_monotonic_ns=stamp,read_started_monotonic_ns=stamp-500_000,
                         image=synthetic_view(self.role,seq))]
    class Model:
        history=deque(maxlen=8)
        def predict_pixels(self,pixels,q):
            self.history.append(q)
            return [.002,.002,-.002,0,0,0,21]
    class Archive(FrameArchive):
        def predict(self,model,pair,pixels,query,stamp):
            if case=="capture_error" and query["mode_at_query"]=="execute":
                from types import SimpleNamespace
                with patch("capture.shutil.disk_usage",return_value=SimpleNamespace(free=0)):
                    return super().predict(model,pair,pixels,query,stamp)
            return super().predict(model,pair,pixels,query,stamp)
    with Archive(directory) as archive:
        runner.infer_loop(link,[Camera(0),Camera(1)],Model(),archive)


def check_capture(log_path,rows):
    import cv2
    import numpy as np
    from policy import prepare_views
    root=log_path.parent/rows[0]["capture_dir"]
    samples=[json.loads(line) for line in (root/"inputs.jsonl").read_text().splitlines()]
    by_stamp={s["stamp"]:s for s in samples}
    assert samples and len(samples)==len(by_stamp)
    assert list(by_stamp)==sorted(by_stamp)
    assert samples[0]["history"]==[samples[0]["stamp"]]*8
    for s in samples:
        stamp=s["stamp"]
        cams=[s["cameras"][key] for key in ("external","wrist")]
        assert stamp==max(c["host_monotonic_ns"] for c in cams)
        assert 0<=s["skew_ns"]<=20_000_000 and stamp<=s["predicted_ns"]
        assert 0<=stamp-min(s["feedback_ns"].values())<=60_000_000 and max(s["feedback_ns"].values())<=stamp
        assert len(s["history"])==8 and s["history"][-1]==stamp and s["history"]==sorted(s["history"])
        assert all(h in by_stamp for h in s["history"])
        inputs=[synthetic_view(i,c["sequence"]) for i,c in enumerate(cams)]
        expected=prepare_views(inputs)
        stored=cv2.imread(str(root/s["images"]["model"]))[:,:,::-1]
        assert np.array_equal(stored,np.concatenate(expected,axis=1))
        scene=cv2.imread(str(root/s["images"]["scene"]))
        assert scene.shape==(480,1280,3)
        for i in range(2):
            assert abs(scene[370,640*i+500].astype(float)-inputs[i][370,500]).max()<5
    joined=[]
    for row in rows:
        if row["event"] not in ("prediction","send_interrupted"): continue
        s=by_stamp[row["stamp"]]
        assert all(row[k]==s[k] for k in ("q","p","images","feedback_ns"))
        assert len(set(s["history"]))==8 and s["predicted_ns"]<=row["mono_ns"]
        if row.get("sent"):
            assert row["mode"]=="execute" and row["sent_ids"]==[0x155,0x156,0x157,0x159]
            assert max(row["measured_at_send_ns"].values())<=row["mono_ns"]<=row["send_started_ns"]<=row["send_finished_ns"]
        joined.append(row)
    assert joined
    return dict(indexed=len(samples),preview=sum(r["mode"]=="preview" for r in joined),
                execute=sum("sent" in r for r in joined),partial=sum(r["event"]=="send_interrupted" for r in joined))


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
            try: tx.send(bus.WIRE.pack(cid,8,data))
            except ConnectionRefusedError:
                # The guard may close its receive socket before the test UI reads "done".
                if rx.fileno()!=-1: raise
                close.set()
        class SimBus(bus.Bus):
            def __init__(self,port):
                self.sock=rx
                self.offset=time.time_ns()-time.monotonic_ns()
            def send(self,cid,data):
                if cid==0x156 and case=="partial": raise OSError("模拟第二条关节报文发送失败")
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
        def mock_worker(link,directory):
            try:
                if case in ("preview","success","partial","capture_error"):
                    try: camera_worker(link,directory,case)
                    except OSError as e:
                        if case=="capture_error": runner.send(link,dict(op="error",text=str(e)))
                        else: raise
                    return
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
                self.thread=threading.Thread(target=mock_worker,args=(self.link,args[-1]))
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
        path=next((folder/"policy_runs").glob("*.jsonl"))
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        last=rows[-1]
        print(case,last,flush=True)
        if case in ("preview","leader"):
            assert not sent and last["outcome"]=="preview" and denied==(case=="leader")
        else:
            assert sent[0][0]==0x151 and sent[-1][0]==0x150 and last["outcome"]==("success" if case=="success" else "aborted"),last
            assert all(i in {0x151,0x155,0x156,0x157,0x159,0x150} for i,_ in sent)
            if case not in ("partial","capture_error"):
                assert any(r.get("mode")=="execute" for r in rows)
            if case=="capture_error":
                assert "128 MiB" in last["detail"] and [cid for cid,_ in sent]==[0x151,0x150]
            if case=="partial":
                row=next(r for r in rows if r["event"]=="send_interrupted")
                assert row["sent_ids"]==[0x155] and "sent" not in row and row["send_failed_ns"]>=row["send_started_ns"]
            assert "已收到从臂 STOP" in last["detail"],last
            if case=="heartbeat": assert "500 ms" in last["detail"],last
            if case=="conflict": assert "其他" in last["detail"],last
        capture=check_capture(path,rows) if case in ("preview","success","partial","capture_error") else None
        return dict(case=case,outcome=last["outcome"],stop_confirmed=stopped.is_set(),simulated_frames=len(sent),capture=capture)


def verify(work):
    import cv2
    import numpy as np
    import torch
    from policy import Predictor,BEST,prepare_views
    from capture import FrameArchive
    torch.set_num_threads(4)
    tests=[simulated_session(c) for c in ("preview","leader","success","partial","capture_error","conflict","heartbeat")]
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
    model.task=0
    model.tag.fill_(0)
    model.history.clear()
    ep=next(e for e in episodes if e["task"]==0 and e["split"]=="val")
    ids=list(range(len(ep["q"])//2-7,len(ep["q"])//2+1))
    inputs=[[cv2.imread(str(view[i])) for view in ep["cameras"]] for i in ids]
    latencies=[]
    with tempfile.TemporaryDirectory(dir=work) as td:
        root=Path(td)/"pairs"
        with FrameArchive(root) as archive:
            for n in range(64):
                idx=n%8
                if n==32: model.history.clear()
                stamp=time.monotonic_ns()
                pair=[dict(image=im,sequence=n+1,host_monotonic_ns=stamp,read_started_monotonic_ns=stamp) for im in inputs[idx]]
                query=dict(q=ep["q"][ids[idx]].tolist(),feedback_ns=dict(joint_first_ns=stamp,joint_last_ns=stamp,gripper_ns=stamp),mode_at_query="preview")
                result,_=archive.predict(model,pair,prepare_views(inputs[idx]),query,stamp)
                latencies.append((time.monotonic_ns()-stamp)/1e6)
        records=[json.loads(l) for l in (root/"inputs.jsonl").read_text().splitlines()]
        assert len(records)==64
        assert records[32]["history"]==[records[32]["stamp"]]*8
        assert records[-1]["history"]==[r["stamp"] for r in records[-8:]]
        model.history.clear()
        for i,s in enumerate(records[-8:]):
            saved=cv2.imread(str(root/s["images"]["model"]))[:,:,::-1]
            restored=np.stack(np.split(saved,2,axis=1))
            assert np.array_equal(restored,prepare_views(inputs[i]))
            replay=model.predict_pixels(restored,s["q"])
        assert np.array_equal(replay,result)
        stored_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    timing=dict(pairs=len(records),archive_bytes=stored_bytes,model_replay_exact=True,
                inference_and_save_p50_ms=round(float(np.median(latencies)),2),
                inference_and_save_p95_ms=round(float(np.percentile(latencies,95)),2),
                inference_and_save_max_ms=round(max(latencies),2))
    return dict(checkpoint=BEST,simulated=tests,replay=samples,capture_replay=timing)


if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve()),allow_nan=False))
