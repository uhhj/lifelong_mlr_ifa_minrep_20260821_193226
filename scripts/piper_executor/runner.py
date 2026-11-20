"""The UI and GPU communicate with an independent actuator/watchdog process."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time

from bus import Bus, NS, State, bounded_target, target_frames

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
SYNC = WORK.parent/"synchronized_capture"
TASKS = ("B1 左侧蓝碟抓放", "B2 右侧浅色碟抓放", "B3 推入左侧黄色区")
SAFETY_NS = 500_000_000
FRESH_NS = 100_000_000


def send(sock, msg):
    sock.sendall(json.dumps(msg, allow_nan=False,ensure_ascii=False).encode())


def receive(sock):
    data = sock.recv(8192)
    if not data: raise EOFError("进程通信已断开")
    return json.loads(data)


def next_pair(cameras, previous):
    groups = [[f for f in cam.frames() if f["sequence"]>seq]
              for cam,seq in zip(cameras,previous)]
    for a in reversed(groups[0]):
        if groups[1]:
            b = min(groups[1],key=lambda f: abs(f["host_monotonic_ns"]-a["host_monotonic_ns"]))
            if abs(a["host_monotonic_ns"]-b["host_monotonic_ns"])<=20_000_000:
                return a,b
    return None


def infer(link, task):
    sys.path.insert(0,str(SYNC))
    from cameras import CameraStream, configured_cameras
    from policy import Predictor
    import torch
    torch.set_num_threads(4)
    model=Predictor(WORK,task)
    with ExitStack() as clean:
        conf=configured_cameras()
        cameras=[]
        for role in ("external","wrist"):
            cam=CameraStream(conf[role])
            clean.callback(cam.close)
            cameras.append(cam)
        previous=(0,0)
        last=0
        last_pair=time.monotonic()
        while True:
            pair=next_pair(cameras,previous)
            if not pair:
                if time.monotonic()-last_pair>3:
                    raise RuntimeError("双相机连续 3 秒无同步图像，请关闭占用相机的录制台")
                time.sleep(.002)
                continue
            last_pair=time.monotonic()
            previous=tuple(f["sequence"] for f in pair)
            stamp=max(f["host_monotonic_ns"] for f in pair)
            if not 0<=time.monotonic_ns()-stamp<=FRESH_NS:
                continue
            send(link,dict(op="query",stamp=stamp))
            q=receive(link)["q"]
            if q is None:
                model.history.clear()
                continue
            if stamp-last>70_000_000: model.history.clear()
            last=stamp
            pred=model.predict([f["image"] for f in pair],q)
            send(link,dict(op="predict",stamp=stamp,q=q,p=pred,full=len(model.history)==8))


class Session:
    def __init__(self, state, send_frame, log, now):
        self.state,self.send_frame,self.log=state,send_frame,log
        self.beat=self.predicted=now
        self.latest=None
        self.requested=0
        self.last_sample=0
        self.started=0
        self.last_sent=0
        self.previous=None

    def watchdog(self,now):
        if not self.state.active: return
        if now-self.beat>=SAFETY_NS or now-self.predicted>=SAFETY_NS:
            raise RuntimeError("窗口心跳或预测超时 500 ms")
        if now-self.started>=30*NS: raise RuntimeError("达到本轮 30 秒时限")
        self.state.ready(now)

    def begin(self,now):
        self.state.ready(now)
        if (not self.latest or not self.latest["full"] or not 0<=now-self.latest["stamp"]<=FRESH_NS
                or now-self.beat>=SAFETY_NS):
            raise ValueError("请等待连续八帧的新鲜预测和窗口心跳")
        self.previous=self.state.observation(now)
        self.started=self.last_sent=now
        self.state.active=True
        self.log(dict(event="start",mono_ns=now,initial=self.previous))
        self.send_frame(0x151,bytes((1,1,10,0,0,0,0,0)))

    def prediction(self,msg,now):
        stamp=msg["stamp"]
        if (stamp!=self.requested or stamp<=self.last_sample or
                not 0<=now-stamp<=FRESH_NS or not msg["full"]):
            self.latest=None
            if self.state.active: raise ValueError("实时预测过期或八帧历史中断")
            return
        self.last_sample=stamp
        self.predicted=now
        self.latest=msg
        row=dict(event="prediction",mono_ns=now,mode="preview",**msg)
        if self.state.active:
            self.watchdog(now)
            if stamp < self.started:
                return
            measured=self.state.observation(now)
            target=bounded_target(msg["p"],measured,self.previous,(now-self.last_sent)/NS)
            for cid,data in target_frames(target): self.send_frame(cid,data)
            self.previous=target
            self.last_sent=now
            row.update(mode="execute",sent=target,measured_at_send=measured)
        self.log(row)


def guard(ui, task, python):
    sys.path.insert(0,str(WORK))
    from stop_link import check_path
    with ExitStack() as clean:
        lock=clean.enter_context((SYNC/"recorder.lock").open("a"))
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("相机已被录制台或另一执行窗口占用")
        check_path(allow_listen_only=True)
        bus=Bus("can0")
        clean.callback(bus.sock.close)
        now=time.monotonic_ns()
        state=State(now)
        logs=WORK/"policy_runs"
        logs.mkdir(mode=0o700,exist_ok=True)
        path=logs/f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_B{task+1}.jsonl"
        file=clean.enter_context(path.open("x",encoding="utf8",buffering=1))
        def log(row):
            file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        session=Session(state,bus.send,log,now)
        log(dict(event="preview",task=TASKS[task],wall_ns=time.time_ns(),mono_ns=now))
        a,b=socket.socketpair(type=socket.SOCK_SEQPACKET)
        clean.callback(a.close)
        with b:
            proc=subprocess.Popen([python,"-B",str(HERE/"runner.py"),"infer",str(b.fileno()),str(task)],
                                  pass_fds=(b.fileno(),),env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"))
        def reap():
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        clean.callback(reap)
        result="已结束只读预测"
        outcome="aborted"
        last_display=0
        last_wait=0
        ui.setblocking(False)
        try:
            send(ui,dict(text="正在载入模型，等待双相机及从臂反馈",log=path.name))
            while True:
                ready=select.select([ui,a,bus.sock],[],[],.015)[0]
                bus.drain(state)
                now=time.monotonic_ns()
                session.watchdog(now)
                if ui in ready:
                    cmd=receive(ui)["op"]
                    if cmd=="beat": session.beat=now
                    elif cmd=="start" and not state.active:
                        try:
                            check_path(allow_listen_only=False)
                            bus.drain(state)
                            session.begin(time.monotonic_ns())
                        except (ValueError,RuntimeError) as e:
                            if state.active: raise
                            send(ui,dict(error=str(e)))
                        else: send(ui,dict(text="执行中",active=True))
                    elif cmd in ("stop","success","failed","close"):
                        if state.active:
                            outcome={"success":"success","failed":"failed"}.get(cmd,"aborted")
                            result="现场结束执行"
                        break
                if a in ready:
                    msg=receive(a)
                    if msg["op"]=="error": raise RuntimeError(msg["text"])
                    if msg["op"]=="query":
                        stamp=msg["stamp"]
                        session.requested=stamp
                        q=state.observation(stamp) if 0<=now-stamp<=FRESH_NS else None
                        send(a,dict(q=q))
                        if q is None and not state.active and now-last_wait>NS:
                            send(ui,dict(text="已收到双相机画面，等待同期的六轴和夹爪反馈"))
                            last_wait=now
                    elif msg["op"]=="predict":
                        session.prediction(msg,time.monotonic_ns())
                        if session.latest and now-last_display>=200_000_000:
                            send(ui,dict(pred=msg["p"],q=msg["q"],active=state.active))
                            last_display=now
                if not state.active and now-session.beat>2*NS: raise RuntimeError("窗口已断连")
        except BaseException as e:
            result=str(e) or type(e).__name__
        finally:
            was_active=state.active
            if was_active:
                state.active=False
                try:
                    stop_at=time.monotonic_ns()
                    bus.send(0x150,bytes((1,0,0,0,0,0,0,0)))
                    deadline=stop_at+SAFETY_NS
                    while time.monotonic_ns()<deadline:
                        bus.drain(state)
                        if state.status and state.status[0]>=stop_at and state.status[1][1]==1: break
                        time.sleep(.003)
                    else: raise RuntimeError("未收到急停反馈")
                    result+="；已收到从臂 STOP 反馈"
                except Exception as e:
                    result+=f"；STOP 未确认，请现场立即使用独立急停：{e}"
                    outcome="aborted"
            log(dict(event="end",outcome=outcome if was_active else "preview",detail=result,mono_ns=time.monotonic_ns()))
            try: send(ui,dict(done=True,text=result))
            except OSError: pass


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("role",choices=["guard","infer"])
    parser.add_argument("fd",type=int)
    parser.add_argument("task",type=int,choices=range(3))
    parser.add_argument("python",nargs="?")
    args=parser.parse_args()
    with socket.socket(fileno=args.fd) as link:
        try:
            if args.role=="infer": infer(link,args.task)
            else: guard(link,args.task,args.python)
        except Exception as e:
            try: send(link,dict(op="error",text=str(e),done=True))
            except OSError: pass
            raise
