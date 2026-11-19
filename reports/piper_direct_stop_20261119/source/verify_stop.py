"""Offline button and kernel-datagram tests, without opening any CAN interface."""
from pathlib import Path
from unittest.mock import patch
import socket
import tempfile
import time

from PyQt5 import QtWidgets
import direct_can
import stop_link
from stop_panel import FRAME, STAMP, STATUS, STOP, TIMESPEC, Panel, StopChannel

EXPECTED=bytes.fromhex('50010000080000000100000000000000')


class Wire:
    def __init__(self):
        self.tx=[]
        self.rx=[]
        self.closed=False
        self.fail=False

    def send(self,packet):
        if self.fail:
            raise BlockingIOError('offline queue full')
        self.tx.append(packet)
        return len(packet)

    def frame(self,stamp,code=1,cid=STATUS,flags=0):
        ts=TIMESPEC.pack(*divmod(stamp,10**9))
        self.rx.append((FRAME.pack(cid,8,bytes([1,code,1,0,0,0,0,0])),
                        [(socket.SOL_SOCKET,STAMP,ts)],flags,None))

    def recvmsg(self,*_):
        if self.rx:
            return self.rx.pop(0)
        raise BlockingIOError()

    def close(self):self.closed=True


def qt_paths():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    wire=Wire()
    events=[]
    pane=Panel(lambda:wire,lambda *row:events.append(row),auto_probe=False)
    pane.show()
    app.processEvents()
    pane.link_result(None,'硬件只听')
    pane.stop.click()
    assert not pane.stop.isEnabled() and not wire.tx
    pane.link_result({'ifindex':26},'')
    pane.tick()
    assert pane.stop.isEnabled() and not wire.tx
    pane.stop.click()
    c=pane.channel
    assert wire.tx==[EXPECTED] and not c.confirmed
    wire.frame(c.first_submit-1)
    wire.frame(time.time_ns(),flags=socket.MSG_DONTROUTE)
    wire.frame(time.time_ns(),cid=0x2a2)
    wire.frame(time.time_ns(),code=0)
    pane.tick()
    assert not c.confirmed
    wire.frame(time.time_ns())
    pane.tick()
    assert c.confirmed and '已进入急停' in pane.status.text()
    wire.frame(time.time_ns(),code=0)
    pane.tick()
    assert not c.confirmed and c.requested is None and wire.tx==[EXPECTED]
    pane.link_result(None,'接口移除')
    pane.stop.click()
    pane.close()
    assert wire.closed and wire.tx==[EXPECTED]


def retry_paths():
    wire=Wire()
    channel=StopChannel(wire,lambda *_:None)
    with patch('stop_panel.time.monotonic') as tick:
        tick.return_value=10.
        channel.request()
        for t in (10.11,10.22,10.35,10.61,12.):
            tick.return_value=t
            channel.poll()
        assert wire.tx==[EXPECTED]*3 and channel.timed_out and not channel.confirmed
        channel.request()
        wire.frame(time.time_ns())
        channel.poll()
        assert channel.confirmed and len(wire.tx)==4
    wire.fail=True
    channel.request()
    wire.frame(time.time_ns())
    channel.poll()
    assert channel.first_submit is None and not channel.confirmed


def kernel_path():
    x,y=socket.socketpair(type=socket.SOCK_DGRAM)
    try:
        x.setblocking(False)
        x.setsockopt(socket.SOL_SOCKET,STAMP,1)
        ch=StopChannel(x,lambda *_:None)
        ch.request()
        assert y.recv(16)==EXPECTED
        y.send(FRAME.pack(STATUS,8,bytes([1,1,1,0,0,0,0,0])))
        ch.poll()
        assert ch.confirmed
    finally:
        x.close();y.close()


def identity_and_capture():
    ident={'ifindex':26,'port':'can0'}
    with tempfile.TemporaryDirectory() as tmp,patch.object(stop_link,'BINDING',Path(tmp)/'id') as file:
        with patch.object(stop_link,'snapshot',return_value=(ident,True)):
            stop_link.check_path(allow_listen_only=True)
            try:stop_link.check_path(allow_listen_only=False)
            except RuntimeError:pass
            else:raise AssertionError('STOP accepted listen-only')
        with patch.object(stop_link,'snapshot',return_value=(ident,False)):
            for contents in ('{}','{"ifindex":25,"port":"can0"}'):
                file.write_text(contents)
                try:direct_can.check_capture_link('can0')
                except RuntimeError:pass
                else:raise AssertionError('accepted unverified normal-mode adapter')
            file.write_text('{"ifindex":26,"port":"can0"}')
            stop_link.check_path(allow_listen_only=False)
            direct_can.check_capture_link('can0')
    tail=direct_can.DirectTail('can0')
    tail.sock=Wire()
    tail.offset=time.time_ns()-time.monotonic_ns()
    tail.sock.rx=[(STOP,[],socket.MSG_DONTROUTE,None)]
    try:list(tail.read(10))
    except RuntimeError as e:assert 'STOP' in str(e)
    else:raise AssertionError('recording ignored local STOP')


if __name__=='__main__':
    assert STOP==EXPECTED
    qt_paths()
    retry_paths()
    kernel_path()
    identity_and_capture()
    print('OFFLINE_OK exact STOP only; GUI open/close silent; timeout/retry; stale/local frames; fresh kernel confirmation; interface identity; capture STOP echo')
