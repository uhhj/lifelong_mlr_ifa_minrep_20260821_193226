"""Onsite PiPER STOP and operator-confirmed reset of an isolated follower."""
from pathlib import Path
import json
import socket
import struct
import time

from PyQt5 import QtCore, QtWidgets
from stop_link import PORT, check_path

FRAME = struct.Struct('=IB3x8s')
STOP = FRAME.pack(0x150,8,bytes.fromhex('0100000000000000'))
RESET = FRAME.pack(0x150, 8, bytes.fromhex('0200000000000000'))
STATUS = 0x2a1
# Motion on the direct bus, including STOP/reset from other programs and role/drive changes.
CONTROL_IDS = (*range(0x150, 0x160), 0x470, 0x471)
STAMP = getattr(socket,'SO_TIMESTAMPNS',35)
TIMESPEC = struct.Struct('@ll')


def open_socket():
    can = socket.socket(socket.PF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
    try:
        filters = b''.join(struct.pack('=II', cid, 0xC00007FF) for cid in (STATUS, *CONTROL_IDS))
        can.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
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
        self.status_stamp = 0
        self.status_error = None
        self.drained = False
        self.control_seen = time.monotonic()
        self.reset_stamp = None
        self.reset_deadline = 0
        self.reset_note = ''
        self.confirm_generation = 0

    def request(self):
        self.cancel_reset('已取消等待复位反馈，STOP 已优先处理' if self.reset_stamp is not None else '')
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

    def cancel_reset(self, note):
        self.confirm_generation += 1
        if self.reset_stamp is not None:
            self.log('reset_confirmation_canceled', note)
        self.reset_stamp = None
        self.reset_note = note

    def reset_blocker(self):
        if self.reset_stamp is not None:
            return '复位已提交，等待新的从臂反馈'
        if self.requested is not None and not self.confirmed and not self.timed_out:
            return '正在等待本次 STOP 的应答'
        if not self.drained or not 0 <= time.time_ns() - self.status_stamp <= 250_000_000:
            return '等待新的从臂状态反馈'
        if self.last_status != 1 or self.status_error != 0:
            return '仅在从臂报告软件急停且无其他错误时可复位'
        if time.monotonic() - self.control_seen < 1.0:
            return '刚收到控制报文，请核对主臂已经物理隔离'
        return ''

    def submit_reset(self, generation):
        self.poll()
        if generation != self.confirm_generation or self.reset_blocker():
            return False
        self.confirm_generation += 1
        try:
            if self.can.send(RESET) != len(RESET):
                raise OSError('复位帧提交不完整')
        except OSError as exc:
            self.reset_note = '复位发送失败：' + str(exc)
            self.log('reset_send_failed', str(exc))
            return False
        self.reset_stamp = time.time_ns()
        self.reset_deadline = time.monotonic() + 1.0
        self.reset_note = '复位已提交，等待新的从臂反馈'
        self.log('reset_submitted', self.reset_stamp)
        return True

    def poll(self):
        self.drained = False
        if self.reset_stamp is not None and time.monotonic() >= self.reset_deadline:
            self.cancel_reset('复位结果未知，未在时限内收到正常状态')
        for _ in range(128):
            try:
                raw, ancillary, flags, _ = self.can.recvmsg(FRAME.size,128)
            except BlockingIOError:
                self.drained = True
                break
            if flags & (socket.MSG_TRUNC|socket.MSG_CTRUNC) or len(raw)!=FRAME.size:
                self.cancel_reset('收到不完整报文，请重新核实状态')
                continue
            cid, size, data = FRAME.unpack(raw)
            if cid in CONTROL_IDS:
                self.control_seen = time.monotonic()
                if self.reset_stamp is not None:
                    self.cancel_reset('观察到其他控制指令，复位结果未知')
                else:
                    self.confirm_generation += 1
                continue
            if flags & socket.MSG_DONTROUTE or cid != STATUS or size != 8:
                continue
            kernel = None
            for kind, code, value in ancillary:
                if kind == socket.SOL_SOCKET and code == STAMP and len(value) >= TIMESPEC.size:
                    sec, ns = TIMESPEC.unpack_from(value)
                    kernel = sec*1_000_000_000+ns
            if kernel is None or kernel < self.status_stamp or not 0 <= time.time_ns()-kernel <= 250_000_000:
                continue
            old_eligible = self.last_status == 1 and self.status_error == 0
            self.status_stamp = kernel
            self.status_error = int.from_bytes(data[6:8], 'big')
            if old_eligible and (data[1] != 1 or self.status_error):
                self.confirm_generation += 1
            self.last_status = data[1]
            self.last_seen = time.monotonic()
            if self.reset_stamp is not None and kernel > self.reset_stamp:
                if self.status_error or self.last_status not in (0, 1):
                    self.cancel_reset('从臂报告异常，复位未确认')
                elif self.last_status == 0:
                    self.reset_stamp = None
                    self.reset_note = '从臂已反馈正常，软件急停已解除'
                    self.log('arm_reset_confirmed', kernel)
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
        self.path_checked = None
        self.setWindowTitle('PiPER 操作臂软件急停与复位')
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
        self.reset = QtWidgets.QPushButton('解除软件急停（复位）')
        self.reset.setEnabled(False)
        self.reset.clicked.connect(self.begin_reset)
        layout.addWidget(self.reset)
        self.reset_status = QtWidgets.QLabel('等待接口核实')
        self.reset_status.setWordWrap(True)
        layout.addWidget(self.reset_status)
        self.confirm_box = QtWidgets.QGroupBox('现场复位确认')
        confirm_layout = QtWidgets.QVBoxLayout(self.confirm_box)
        self.attest = QtWidgets.QCheckBox('我已确认主臂与从臂的 CAN 物理断开，\n从臂已可靠支撑，人员已退出运动范围。')
        confirm_layout.addWidget(self.attest)
        actions = QtWidgets.QHBoxLayout()
        self.apply_reset = QtWidgets.QPushButton('确认复位')
        self.apply_reset.setEnabled(False)
        self.apply_reset.clicked.connect(self.finish_reset)
        self.attest.toggled.connect(self.apply_reset.setEnabled)
        cancel = QtWidgets.QPushButton('取消')
        cancel.clicked.connect(self.dismiss_reset)
        for button in (self.reset, self.apply_reset, cancel):
            button.setAutoDefault(False)
        actions.addWidget(cancel)
        actions.addWidget(self.apply_reset)
        confirm_layout.addLayout(actions)
        layout.addWidget(self.confirm_box)
        self.confirm_box.hide()
        self.consent = None
        self.consent_expires = 0
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
        self.dismiss_reset()
        self.reset.setEnabled(False)
        self.reset_status.setText('发送接口不可用')
        if self.channel:
            self.channel.can.close()
        self.identity = self.channel = None
        self.path_checked = None
        self.stop.setEnabled(False)

    def link_result(self, identity, error):
        if error or identity!=self.identity:
            self.detach()
        if error:
            self.status.setText('STOP 和复位不可用：'+error)
            self.feedback.setText('停机状态未确认')
            return
        self.path_checked = time.monotonic()
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
        self.dismiss_reset()
        if self.channel:
            self.channel.request()
            self.status.setText(self.channel.message)
            self.show_reset_state()

    def dismiss_reset(self):
        self.consent = None
        self.attest.setChecked(False)
        self.confirm_box.hide()

    def begin_reset(self):
        self.tick()
        ch = self.channel
        if ch is None or self.reset_reason() or self.consent is not None:
            return
        self.consent = (ch, ch.confirm_generation)
        self.consent_expires = time.monotonic() + 30
        self.attest.setChecked(False)
        self.confirm_box.show()
        self.show_reset_state()

    def finish_reset(self):
        consent = self.consent
        authorized = self.attest.isChecked() and time.monotonic() < self.consent_expires
        self.dismiss_reset()
        if consent is None or not authorized or self.channel is not consent[0] or self.reset_reason():
            return
        try:
            self.channel.submit_reset(consent[1])
        except OSError as exc:
            self.link_result(None, str(exc))
            return
        self.show_reset_state()

    def reset_reason(self):
        if self.path_checked is None or time.monotonic() - self.path_checked > 1.5:
            return '等待接口身份核实'
        return self.channel.reset_blocker()

    def show_reset_state(self):
        ch = self.channel
        reason = self.reset_reason()
        if self.consent is not None and (
            reason or self.consent != (ch, ch.confirm_generation) or time.monotonic() >= self.consent_expires
        ):
            self.dismiss_reset()
        self.reset.setEnabled(not reason and self.consent is None)
        if self.consent is not None:
            self.reset_status.setText('复位可能导致手臂下落。确认后发送一次。')
        else:
            note = ch.reset_note
            if reason and reason != note:
                note = (note + '\n' + reason).strip()
            self.reset_status.setText(note or '复位仅用于单独连接、已得到支撑的从臂。')

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
        self.show_reset_state()

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
