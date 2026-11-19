"""Receive native PiPER commands and follower feedback on one listen-only CAN interface."""
from collections import deque
import socket
import struct
import subprocess
import time

from timeline import JOINT_RX, JOINT_TX, Timeline, decode_joints, last_at

IDS = (0x150, 0x151, *JOINT_TX, 0x159, 0x2A1, *JOINT_RX, 0x2A8, 0x470, 0x471)
WIRE = struct.Struct('=IB3x8s')
TIMESPEC = struct.Struct('@ll')
TIMESTAMP = getattr(socket, 'SO_TIMESTAMPNS', 35)
OVERFLOW = getattr(socket, 'SO_RXQ_OVFL', 40)
GROUP_NS = 3_000_000


def check_listen_only(port):
    ip = ['/usr/sbin/ip', '-details', 'link', 'show', 'dev', port]
    report = subprocess.run(ip, capture_output=True, text=True, check=True).stdout
    flags = report.split('<', 1)[1].split('>', 1)[0].split(',') if '<' in report else []
    if ('UP' not in flags or 'LOWER_UP' not in flags or
            not any('LISTEN-ONLY' in s for s in report.splitlines() if s.strip().startswith('can ')) or
            'bitrate 1000000 ' not in report):
        raise RuntimeError(f'{port} 需要处于 UP、1 Mbit/s、硬件 LISTEN-ONLY 才能旁听官方直连')


class DirectTimeline(Timeline):
    def __init__(self):
        super().__init__()
        self.pending = {JOINT_TX: [], JOINT_RX: []}
        self.modes = deque(maxlen=512)
        self.last_state = None

    def ingest_bus(self, cid, data, stamp):
        if len(data) != 8:
            raise ValueError(f'直连 CAN 报文长度异常 {cid:#x}')
        if stamp < self.watermark:
            raise ValueError('直连 CAN 内核时间戳倒退')
        gap = self.watermark and stamp-self.watermark > 250_000_000
        state_gap = cid == 0x2A1 and self.last_state is not None and stamp-self.last_state > 250_000_000
        if gap or state_gap:
            # A brief power interruption may not remove the host interface.
            self.targets.append(dict(mono_ns=stamp,valid=False))
            self.target_grip.append(dict(mono_ns=stamp,valid=False))
            self.modes.append(dict(mono_ns=stamp,value=None))
            for pending in self.pending.values():
                pending.clear()
        if cid == 0x2A1:
            self.last_state = stamp
        self.watermark = stamp
        for ids, packet in self.pending.items():
            if packet and stamp-packet[0][0] > GROUP_NS:
                # Invalidate a held command if a new joint group was only partly observed.
                if ids == JOINT_TX:
                    self.targets.append(dict(mono_ns=packet[0][0], valid=False))
                packet.clear()
        if cid == 0x150 and data[0] != 0:
            raise RuntimeError('直连总线上出现停止或复位指令，结束本条录制')
        if cid == 0x470 or (cid == 0x471 and data[1] == 1):
            raise RuntimeError('直连总线上出现角色配置或失能指令，结束本条录制')
        if cid == 0x2A1 and (data[1] != 0 or data[6:8] != bytes(2)):
            raise RuntimeError(f'操作臂报告故障 {data.hex()}')
        if cid == 0x151:
            if data[:2] != b'\x01\x01' or data[3] != 0xAD:
                raise RuntimeError(f'控制模式离开高跟随 {data.hex()}')
            self.modes.append(dict(mono_ns=stamp, value=data.hex()))
        elif cid in (0x159, 0x2A8):
            queue = self.target_grip if cid == 0x159 else self.measured_grip
            queue.append(dict(mono_ns=stamp, width=struct.unpack_from('>i', data)[0]/1000,
                              frame=data.hex(), valid=cid != 0x159 or (data[6] in (1,3) and data[7]==0)))
        elif cid in JOINT_TX or cid in JOINT_RX:
            ids = JOINT_TX if cid in JOINT_TX else JOINT_RX
            packet = self.pending[ids]
            if cid == ids[0]:
                if packet and ids == JOINT_TX:
                    self.targets.append(dict(mono_ns=packet[0][0], valid=False))
                packet[:] = [(stamp,data)]
            elif packet and cid == ids[len(packet)]:
                packet.append((stamp,data))
                if len(packet) == len(ids):
                    frames = [d for _,d in packet]
                    queue = self.targets if ids == JOINT_TX else self.feedback
                    queue.append(dict(mono_ns=stamp, joints=decode_joints(frames),
                                      component_monotonic_ns=[t for t,_ in packet],
                                      frames=[d.hex() for d in frames], valid=True))
                    packet.clear()
            elif ids == JOINT_TX:
                self.targets.append(dict(mono_ns=packet[0][0] if packet else stamp,valid=False))
                packet.clear()
            else:
                packet.clear()

    def command_at(self, stamp):
        joint = last_at(self.targets, stamp)
        grip = last_at(self.target_grip, stamp)
        if not joint or not joint['valid'] or not grip or not grip['valid']:
            return None
        mode = last_at(self.modes, stamp)
        return dict(joints_rad=joint['joints'], gripper_mm=grip['width'], origin='direct_can_rx',
                    sample_monotonic_ns=max(joint['mono_ns'],grip['mono_ns']),
                    joint_bus_monotonic_ns=joint['component_monotonic_ns'],
                    gripper_bus_monotonic_ns=grip['mono_ns'], control_mode_hex=mode['value'] if mode else None,
                    frame_hex=joint['frames']+[grip['frame']])


class DirectTail:
    def __init__(self, port):
        self.port = port
        self.timeline = DirectTimeline()
        self.sock = None

    def open(self):
        check_listen_only(self.port)
        anchors = []
        for _ in range(7):
            before = time.monotonic_ns()
            wall = time.time_ns()
            after = time.monotonic_ns()
            anchors.append((after-before,wall-(before+after)//2))
        self.offset = min(anchors)[1]
        sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            filters = b''.join(struct.pack('=II', cid, 0xC00007FF) for cid in IDS)
            sock.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,filters)
            sock.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_ERR_FILTER,struct.pack('=I',0x1fffffff))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            sock.setsockopt(socket.SOL_SOCKET,TIMESTAMP,1)
            sock.setsockopt(socket.SOL_SOCKET,OVERFLOW,1)
            sock.bind((self.port,))
            sock.setblocking(False)
            self.sock = sock
        except BaseException:
            sock.close()
            raise

    def read(self, limit):
        if self.sock is None:
            self.open()
        if abs(time.time_ns()-time.monotonic_ns()-self.offset)>20_000_000:
            raise RuntimeError('直连 CAN 时钟基准发生跳变')
        for _ in range(limit):
            try:
                raw,anc,flags,_ = self.sock.recvmsg(WIRE.size,128)
            except BlockingIOError:
                return
            if flags & (socket.MSG_DONTROUTE|socket.MSG_CTRUNC|socket.MSG_TRUNC):
                raise RuntimeError('直连监听发现本机发送回显或截断帧')
            cid,dlc,data=WIRE.unpack(raw)
            if cid & 0x20000000:
                raise RuntimeError(f'直连 CAN 错误帧 {cid:#x}')
            stamp = None
            for level,kind,value in anc:
                if level==socket.SOL_SOCKET and kind==TIMESTAMP:
                    sec,ns=TIMESPEC.unpack_from(value)
                    stamp=sec*1_000_000_000+ns-self.offset
                if level==socket.SOL_SOCKET and kind==OVERFLOW and struct.unpack_from('=I',value)[0]:
                    raise RuntimeError('直连监听接收队列丢帧')
            if stamp is None:
                raise RuntimeError('直连 CAN 缺少内核接收时间')
            self.timeline.ingest_bus(cid,data[:dlc],stamp)
            yield None

    def close(self):
        if self.sock:
            self.sock.close()
