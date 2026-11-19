"""Onsite software stop. The only transmitted CAN payload is the native PiPER STOP."""
from pathlib import Path
import json
import socket
import struct
import threading
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from stop_link import PORT, check_path

FRAME = struct.Struct('=IB3x8s')
STOP = FRAME.pack(0x150,8,bytes.fromhex('0100000000000000'))
STATUS = 0x2a1
STAMP = getattr(socket,'SO_TIMESTAMPNS',35)
TIMESPEC = struct.Struct('@ll')


def open_socket():
    can = socket.socket(socket.PF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
    try:
        can.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,struct.pack('=II',STATUS,0xC00007FF))
        can.setsockopt(socket.SOL_SOCKET,STAMP,1)
        can.bind((PORT,))
        can.setblocking(False)
        return can
    except BaseException:
        can.close()
        raise


class StopChannel:
    def __init__(self, can, log):
        self.can = can
        self.log = log
        self.message = '等待操作臂反馈'
        self.last_status = None
        self.last_seen = None
        self.requested = None
        self.first_submit = None
        self.tries = 0
        self.confirmed = False
        self.timed_out = False

    def request(self):
        self.requested = time.monotonic()
        self.first_submit = None
        self.tries = 0
        self.timed_out = self.confirmed = False
        self.transmit()

    def transmit(self):
        self.tries += 1
        try:
            if self.can.send(STOP) != len(STOP):
                raise OSError('STOP 帧提交不完整')
        except OSError as exc:
            self.message = f'STOP 发送失败：{exc}'
            self.log('send_failed',str(exc))
            return
        if self.first_submit is None:
            self.first_submit = time.time_ns()
        self.message = 'STOP 已提交，等待操作臂反馈'
        self.log('stop_submitted',self.tries)

    def poll(self):
        for _ in range(128):
            try:
                raw, ancillary, flags, _ = self.can.recvmsg(FRAME.size,128)
            except BlockingIOError:
                break
            if flags & (socket.MSG_DONTROUTE|socket.MSG_TRUNC|socket.MSG_CTRUNC) or len(raw)!=FRAME.size:
                continue
            cid, size, data = FRAME.unpack(raw)
            if cid != STATUS or size != 8:
                continue
            kernel = None
            for kind, code, value in ancillary:
                if kind == socket.SOL_SOCKET and code == STAMP:
                    sec, ns = TIMESPEC.unpack_from(value)
                    kernel = sec*1_000_000_000+ns
            if kernel is None or not 0 <= time.time_ns()-kernel <= 250_000_000:
                continue
            self.last_status = data[1]
            self.last_seen = time.monotonic()
            if self.requested is None:
                self.message = '收到操作臂急停状态' if data[1]==1 else '收到操作臂状态反馈'
            elif self.first_submit is not None and kernel >= self.first_submit:
                if data[1]==1 and not self.confirmed and time.monotonic()-self.requested<3:
                    self.confirmed = True
                    self.message = '操作臂反馈：已进入急停状态'
                    self.log('arm_stop_confirmed',kernel)
                elif data[1]!=1 and self.confirmed:
                    self.confirmed = False
                    self.requested = None
                    self.message = '操作臂已退出急停状态'
                    self.log('left_stop_state',kernel)
        if self.requested is None or self.confirmed:
            return
        age = time.monotonic()-self.requested
        if age>=0.6 and not self.timed_out:
            self.timed_out = True
            self.message = '未收到操作臂急停确认，请使用现场实体急停'
            self.log('unconfirmed',self.tries)
        elif age<0.6 and self.tries<3 and age>=self.tries*0.1:
            self.transmit()


class Probe(QtCore.QThread):
    result = QtCore.pyqtSignal(object,str)

    def run(self):
        try:
            self.result.emit(check_path(allow_listen_only=False),'')
        except Exception as exc:
            self.result.emit(None,str(exc))


class Panel(QtWidgets.QWidget):
    def __init__(self, socket_factory=open_socket, log=None, auto_probe=True):
        super().__init__()
        self.factory = socket_factory
        self.log = log or self.write_log
        self.identity = self.channel = None
        self.setWindowTitle('PiPER 操作臂软件急停')
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint,True)
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel('正在核实 can0')
        self.feedback = QtWidgets.QLabel('尚未收到操作臂反馈')
        self.stop = QtWidgets.QPushButton('停止操作臂  STOP')
        self.stop.setMinimumSize(380,115)
        self.stop.setStyleSheet('QPushButton {background:#a50d22;color:white;font-size:30px;font-weight:bold;} QPushButton:disabled {background:#575757;color:white;}')
        self.stop.setEnabled(False)
        self.stop.pressed.connect(self.press)
        layout.addWidget(self.stop)
        layout.addWidget(self.status)
        layout.addWidget(self.feedback)
        layout.addWidget(QtWidgets.QLabel('软件停机后，手臂可能缓慢下落。'))
        self.probe = Probe(self)
        self.probe.result.connect(self.link_result)
        self.probe_timer = QtCore.QTimer(self)
        self.probe_timer.timeout.connect(self.check)
        self.io_timer = QtCore.QTimer(self)
        self.io_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.io_timer.timeout.connect(self.tick)
        self.io_timer.start(15)
        if auto_probe:
            self.probe_timer.start(600)
            self.check()

    def write_log(self,event,detail):
        try:
            with Path(__file__).with_name('onsite_stop.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(time_ns=time.time_ns(),mono_ns=time.monotonic_ns(),
                                        event=event,detail=detail),ensure_ascii=False)+'\n')
        except OSError:
            pass

    def check(self):
        if not self.probe.isRunning():
            self.probe.start()

    def detach(self):
        if self.channel:
            self.channel.can.close()
        self.identity = self.channel = None
        self.stop.setEnabled(False)

    def link_result(self, identity, error):
        if error or identity!=self.identity:
            self.detach()
        if error:
            self.status.setText('STOP 不可用：'+error)
            self.feedback.setText('停机状态未确认')
            return
        if self.channel is None:
            try:
                self.channel=StopChannel(self.factory(),self.log)
            except OSError as exc:
                self.status.setText('无法打开 STOP 接口：'+str(exc))
                return
            self.identity=identity
            self.stop.setEnabled(True)
            self.tick()

    def press(self):
        if self.channel:
            self.channel.request()
            self.status.setText(self.channel.message)

    def tick(self):
        ch=self.channel
        if ch is None:
            return
        try: ch.poll()
        except OSError as exc:
            self.link_result(None,str(exc))
            return
        fresh=ch.last_seen is not None and time.monotonic()-ch.last_seen<=.35
        self.status.setText(ch.message if fresh or ch.requested is not None else '发送通道已打开，尚无新鲜状态反馈')
        self.feedback.setText(f'实时反馈：机械臂状态 0x{ch.last_status:02X}' if fresh else '当前反馈缺失，无法确认当前停机状态')
        if not fresh and ch.confirmed:
            self.status.setText('曾收到急停反馈，当前状态未知')

    def closeEvent(self,event):
        self.probe_timer.stop()
        self.io_timer.stop()
        self.probe.wait()
        self.detach()
        super().closeEvent(event)


if __name__=='__main__':
    app=QtWidgets.QApplication([])
    pane=Panel()
    pane.show()
    app.exec_()
