"""Single-process CAN owner. A preview session never sends on the bus."""
from collections import deque
import math
import socket
import struct
import time

NS = 1_000_000_000
JOINT = (0x2A5, 0x2A6, 0x2A7)
DRIVES = tuple(range(0x261, 0x267))
COMMANDS = {*range(0x150,0x160), *range(0x470,0x472)}
WIRE = struct.Struct("=IB3x8s")
LIMITS = [(-154,154),(0,195),(-175,0),(-106,106),(-75,75),(-100,100)]
TS = getattr(socket, "SO_TIMESTAMPNS", 35)
OVER = getattr(socket, "SO_RXQ_OVFL", 40)


class Bus:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            ids = sorted(COMMANDS | set(JOINT + DRIVES + (0x2A1, 0x2A8)))
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                                 b"".join(struct.pack("=II", i, 0xc00007ff) for i in ids))
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER,struct.pack("=I",0x1fffffff))
            self.sock.setsockopt(socket.SOL_SOCKET, TS, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, OVER, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            # Own sends are excluded; commands from every other socket remain visible.
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
            self.sock.bind((port,))
            self.sock.setblocking(False)
            self.offset = time.time_ns()-time.monotonic_ns()
        except BaseException:
            self.sock.close()
            raise

    def drain(self, state):
        if abs(time.time_ns()-time.monotonic_ns()-self.offset) > 20_000_000:
            raise RuntimeError("CAN 时钟跳变")
        for _ in range(4096):
            try:
                raw, anc, flags, _ = self.sock.recvmsg(WIRE.size, 128)
            except BlockingIOError:
                return
            cid, length, data = WIRE.unpack(raw)
            if length != 8 or cid & 0xe0000000 or flags & (socket.MSG_TRUNC|socket.MSG_CTRUNC):
                raise ValueError("CAN 错误帧或截断")
            stamp = None
            for level, kind, v in anc:
                if level == socket.SOL_SOCKET and kind == TS:
                    sec, ns = struct.unpack_from("@ll",v)
                    stamp = sec*NS+ns-self.offset
                if level == socket.SOL_SOCKET and kind == OVER and struct.unpack_from("=I",v)[0]:
                    raise ValueError("CAN 接收丢帧")
            if stamp is None:
                raise ValueError("缺少 CAN 内核时间")
            state.ingest(cid, data, stamp)
        raise RuntimeError("CAN 处理积压")

    def send(self, cid, data):
        if self.sock.send(WIRE.pack(cid,8,data)) != WIRE.size:
            raise OSError("CAN 发送不完整")


class State:
    def __init__(self, now):
        self.last_other = now
        self.last_seen = 0
        self.pending = []
        self.joints, self.widths = deque(maxlen=400), deque(maxlen=400)
        self.drives = {}
        self.status = None
        self.active = False

    def ingest(self, cid, data, stamp):
        if stamp < self.last_seen:
            raise ValueError("CAN 时间倒退")
        self.last_seen = stamp
        if cid in COMMANDS:
            self.last_other = stamp
            if self.active:
                raise RuntimeError("检测到其他运动、急停或配置指令")
        elif cid == 0x2A1:
            self.status = stamp,data
            if self.active and not self.mode_ok():
                raise RuntimeError("从臂已退出正常 CAN 关节模式")
        elif cid in DRIVES:
            self.drives[cid] = stamp,data[5]
            if self.active and data[5] != 0x40:
                raise RuntimeError(f"关节 {cid-0x260} 驱动状态异常")
        elif cid == 0x2A8:
            if data[7] != 0:
                raise ValueError("夹爪不是毫米行程模式")
            self.widths.append((stamp,struct.unpack_from(">i",data)[0]/1000,data[6]))
            if self.active and data[6] & 0x7f != 0x40:
                raise RuntimeError("夹爪状态异常")
        elif cid in JOINT:
            if cid == JOINT[0]:
                self.pending = [(stamp,data)]
            elif self.pending and cid == JOINT[len(self.pending)] and stamp-self.pending[0][0] <= 3_000_000:
                self.pending.append((stamp,data))
                if len(self.pending)==3:
                    q=[math.radians(n/1000) for _,d in self.pending for n in struct.unpack(">ii",d)]
                    self.joints.append((stamp,self.pending[0][0],q))
                    self.pending = []
            else:
                self.pending = []

    def observation(self, stamp):
        j=next((x for x in reversed(self.joints) if x[0]<=stamp),None)
        g=next((x for x in reversed(self.widths) if x[0]<=stamp),None)
        if j and g and stamp-min(j[1],g[0])<=60_000_000:
            return j[2]+[g[1]]
        return None

    def mode_ok(self):
        if not self.status: return False
        data = self.status[1]
        return data[:4] == bytes((1,0,1,0)) and data[6:8] == bytes(2)

    def ready(self, now):
        if now-self.last_other < NS:
            raise ValueError("有其他控制源，或独立接线观察时间不足 1 秒")
        if not self.mode_ok() or not 0<=now-self.status[0]<=150_000_000:
            raise ValueError("请现场确认从臂已处于正常 CAN / MOVE J 位置速度模式")
        if any(cid not in self.drives or self.drives[cid][1]!=0x40 or
               not 0<=now-self.drives[cid][0]<=500_000_000 for cid in DRIVES):
            raise ValueError("六轴驱动反馈未就绪，请现场核对使能和故障")
        if self.observation(now) is None or self.widths[-1][2]&0x7f != 0x40:
            raise ValueError("从臂位置或夹爪使能反馈未就绪")


def bounded_target(prediction, measured, previous, dt):
    if any(len(a)!=7 or not all(math.isfinite(v) for v in a) for a in (prediction,measured,previous)):
        raise ValueError("七维目标或反馈无效")
    limits=[(math.radians(lo),math.radians(hi)) for lo,hi in LIMITS]+[(0,70)]
    target=[]
    for i, ((lo,hi),p,q,last) in enumerate(zip(limits,prediction,measured,previous)):
        tolerance = math.radians(1) if i<6 else 2
        tracking = math.radians(5) if i<6 else 12
        if not lo-tolerance<=q<=hi+tolerance or abs(q-last)>tracking:
            raise ValueError("跟踪偏差或关节范围异常")
        step=(math.radians(24) if i<6 else 35)*min(max(dt,0),.05)
        target.append(min(hi,max(lo, min(last+step,max(last-step,p)))))
    return target


def target_frames(target):
    joints=[round(math.degrees(x)*1000) for x in target[:6]]
    for i in range(3):
        yield 0x155+i,struct.pack(">ii",*joints[i*2:i*2+2])
    yield 0x159,struct.pack(">iHBB",round(target[6]*1000),1000,1,0)
