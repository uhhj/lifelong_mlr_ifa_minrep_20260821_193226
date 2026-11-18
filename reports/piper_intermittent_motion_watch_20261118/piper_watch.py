"""Passive PiPER observation. Never writes to a CAN socket or changes CAN state."""
import argparse
from collections import Counter
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import struct
import time


PORTS = ('can0', 'can1')
CAN_IDS = (0x150,0x151,0x155,0x156,0x157,0x159,0x2A1,0x2A2,0x2A3,0x2A4,0x2A5,0x2A6,0x2A7,0x2A8,
           0x470,0x471,0x472,0x473,0x475,0x477,0x47E, *range(0x251,0x257), *range(0x261,0x267))
ERROR_FLAG = 0x20000000
SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)
SO_RXQ_OVFL = getattr(socket, 'SO_RXQ_OVFL', 40)
TIMESPEC = struct.Struct('@ll')
UINT = struct.Struct('=I')
WIRE = struct.Struct('=IB3x8s')
RECORD = struct.Struct('<QQIBBH8s')
FILTERS = b''.join(struct.pack('=II', cid, 0xC00007FF) for cid in CAN_IDS)
PART_SIZE = 64 * 1024 * 1024
PARTS = 64
COUNTERS = ('rx_packets','rx_errors','rx_dropped','tx_packets','tx_errors','tx_dropped')
ANCILLARY_SIZE = socket.CMSG_SPACE(TIMESPEC.size) + socket.CMSG_SPACE(UINT.size)


def subscribe(port):
    rx = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        rx.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, FILTERS)
        rx.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER, UINT.pack(0x1FFFFFFF))
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        rx.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        rx.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        rx.bind((port,))
        rx.setblocking(False)
        return rx
    except BaseException:
        rx.close()
        raise


def decode(port, payload, ancillary, flags, receipt):
    cid, dlc, data = WIRE.unpack(payload)
    kernel = overflow = 0
    for level, code, value in ancillary:
        if level != socket.SOL_SOCKET:
            continue
        if code == SO_TIMESTAMPNS and len(value) >= TIMESPEC.size:
            seconds, nanos = TIMESPEC.unpack_from(value)
            kernel = seconds * 1_000_000_000 + nanos
        elif code == SO_RXQ_OVFL and len(value) >= UINT.size:
            overflow, = UINT.unpack_from(value)
    packed = RECORD.pack(kernel, receipt, cid, port, min(dlc, 8), flags & 0xFFFF, data)
    return packed, cid, overflow, kernel


def reading(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def clock_anchor():
    samples = []
    for _ in range(5):
        before = time.monotonic_ns()
        wall = time.time_ns()
        after = time.monotonic_ns()
        samples.append((after-before, wall, (before+after)//2))
    _, wall, mono = min(samples)
    return wall, mono


class Ring:
    def __init__(self, output):
        self.output = output
        self.serial = 0
        self.stream = None
        self.size = 0

    def append(self, items):
        if self.stream is None or self.size+len(items) > PART_SIZE:
            self.close()
            if self.serial >= PARTS:
                (self.output / f'frames-{self.serial-PARTS:08d}.bin').unlink()
            self.stream = (self.output / f'frames-{self.serial:08d}.bin').open('xb', buffering=64*1024)
            self.serial += 1
            self.size = 0
        self.stream.write(items)
        self.size += len(items)

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def run(output, seconds):
    selectorset = selectors.DefaultSelector()
    ring = Ring(output)
    sockets = {}
    counts = {port: Counter() for port in PORTS}
    overflows = Counter()
    generations = Counter()
    up = set()
    active = True
    deadline = time.monotonic() + seconds
    def terminate(_signal, _frame):
        nonlocal active
        active = False
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with (output/'health.jsonl').open('x', encoding='utf-8', buffering=1) as health:
        def log(event, **fields):
            wall, mono = clock_anchor()
            health.write(json.dumps(dict(kind=event, wall_ns=wall, mono_ns=mono, **fields), separators=(',',':'))+'\n')

        def sync_links():
            for name in PORTS:
                index = reading(f'/sys/class/net/{name}/ifindex')
                if name in sockets and sockets[name][0] != index:
                    _, rx = sockets.pop(name)
                    selectorset.unregister(rx)
                    rx.close()
                    log('interface_removed', interface=name)
                if index is not None and name not in sockets:
                    try:
                        rx = subscribe(name)
                    except OSError as exc:
                        log('bind_error', interface=name, errno=exc.errno)
                        continue
                    sockets[name] = (index,rx)
                    selectorset.register(rx,selectors.EVENT_READ,name)
                    generations[name] += 1
                    log('interface_bound', interface=name, generation=generations[name], ifindex=index,
                        receive_buffer=rx.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF))

        def snapshot():
            counters = {port: {name: reading(f'/sys/class/net/{port}/statistics/{name}') for name in COUNTERS}
                        for port in PORTS}
            cpu = reading('/proc/stat') or ''
            disk = reading('/proc/diskstats') or ''
            log('health', bound=sorted(sockets), counts=counts, overflow=overflows, link=counters,
                cpu=cpu.splitlines()[:1], load=reading('/proc/loadavg'),
                disk=[line.strip() for line in disk.splitlines() if line.split()[2] in ('sdb','nvme0n1')],
                pressure={name:reading(f'/proc/pressure/{name}') for name in ('cpu','io','memory')})

        try:
            sync_links()
            if set(sockets) != set(PORTS):
                raise RuntimeError('Both CAN interfaces must bind before readiness')
            start_wall = time.time_ns()
            ready=dict(pid=os.getpid(), interfaces=sorted(sockets), started_wall_ns=start_wall,
                       expires_wall_ns=start_wall+seconds*1_000_000_000, record_layout=RECORD.format,
                       record_size=RECORD.size, directions='MSG_DONTROUTE marks local echo',
                       columns=['kernel_realtime_ns','observed_monotonic_ns','can_id_with_flags','port_index',
                                'length','recv_flags','data'], limit_bytes=PART_SIZE*PARTS,
                       errors_mask='0x1fffffff', error_flag=ERROR_FLAG, timestamp_socket='SO_TIMESTAMPNS',
                       ids=CAN_IDS, receive_only=True)
            (output/'ready.json').write_text(json.dumps(ready)+'\n', encoding='utf-8')
            next_health = next_link = next_flush = 0.0
            previous_overflow = Counter()
            while active and time.monotonic()<deadline:
                batch=bytearray()
                for key,_ in selectorset.select(timeout=.25):
                    rx,name=key.fileobj,key.data
                    index=PORTS.index(name)
                    for _ in range(512):
                        try:
                            data, ancillary, flags, _ = rx.recvmsg(WIRE.size,ANCILLARY_SIZE)
                        except BlockingIOError:
                            break
                        except OSError as exc:
                            if exc.errno not in (errno.ENETDOWN,errno.ENODEV,errno.ENXIO) or name in up:
                                log('socket_error',interface=name,errno=exc.errno)
                            up.discard(name)
                            break
                        if len(data)!=WIRE.size:
                            continue
                        up.add(name)
                        frame,cid,overflow,kernel=decode(index,data,ancillary,flags,time.monotonic_ns())
                        batch.extend(frame)
                        counts[name]['frames']+=1
                        if cid & ERROR_FLAG:
                            counts[name][f'error_{cid & 0x1FFFFFFF:08x}']+=1
                        else:
                            counts[name][f'id_{cid:03x}']+=1
                        if not kernel:
                            counts[name]['timestamp_missing']+=1
                        key_for_overflow=(name,generations[name])
                        if overflow>previous_overflow[key_for_overflow]:
                            overflows[name]+=overflow-previous_overflow[key_for_overflow]
                            previous_overflow[key_for_overflow]=overflow
                if batch:
                    ring.append(batch)
                now=time.monotonic()
                if now>=next_link:
                    sync_links(); next_link=now+1
                if now>=next_flush:
                    ring.flush(); next_flush=now+1
                if now>=next_health:
                    snapshot(); next_health=now+5
            snapshot()
            log('end',reason='deadline' if active else 'signal')
        finally:
            ring.close()
            for _,rx in sockets.values():
                selectorset.unregister(rx)
                rx.close()
            selectorset.close()


def self_test():
    msg = WIRE.pack(ERROR_FLAG|0x244, 8, bytes(range(8)))
    anc = [(socket.SOL_SOCKET,SO_TIMESTAMPNS,TIMESPEC.pack(3,200)),
           (socket.SOL_SOCKET,SO_RXQ_OVFL,UINT.pack(12))]
    packed,cid,drops,stamp = decode(1,msg,anc,socket.MSG_DONTROUTE,5100)
    assert drops==12 and stamp==3_000_000_200
    assert RECORD.unpack(packed)==(stamp,5100,cid,1,8,socket.MSG_DONTROUTE,bytes(range(8)))
    assert all(any(struct.unpack_from('=II',FILTERS,i)==(c,0xC00007FF) for i in range(0,len(FILTERS),8))
               for c in (*range(0x251,0x257),*range(0x261,0x267),0x150,0x156))
    print('passive frame schema and filters verified')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--seconds',type=int,default=86400)
    parser.add_argument('--self-test',action='store_true')
    options=parser.parse_args()
    if options.self_test:
        self_test()
    else:
        run(options.output,options.seconds)
