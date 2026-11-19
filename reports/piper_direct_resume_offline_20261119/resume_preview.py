"""Offline Qt preview of direct-teleop recovery. All feedback and actions stay in memory."""
import argparse
from collections import deque
import socket
import sys
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'piper_direct_stop_20261119' / 'source'))
from stop_panel import FRAME, Panel, STAMP, STATUS, STOP, TIMESPEC


class SimulatedBus:
    """The operator panel's original STOP socket is replaced with this memory queue."""

    def __init__(self, handoff):
        self.handoff = handoff  # Hypothetical positive interlock, never inferred from traffic silence.
        self.aligned = True
        self.online = True
        self.fault = False
        self.status = 0
        self.enabled = [True] * 6
        self.actions = []
        self.rx = deque()
        self.closed = False

    def send(self, packet):
        if self.closed or packet != STOP:
            raise OSError('模拟接口仅接收原窗口的 STOP')
        self.actions.append('STOP')
        self.status = 1
        return len(packet)

    def sample(self):
        if self.online and not self.closed:
            wall = time.time_ns()
            data = bytes([0, self.status, 0, 0, 0, 0, 0, int(self.fault)])
            stamp = TIMESPEC.pack(*divmod(wall, 10**9))
            self.rx.append((FRAME.pack(STATUS, 8, data), [(socket.SOL_SOCKET, STAMP, stamp)], 0, None))
            return time.monotonic(), tuple(self.enabled)
        return None, ()

    def recvmsg(self, *_):
        if self.closed:
            raise OSError('模拟接口已关闭')
        if not self.rx:
            raise BlockingIOError()
        return self.rx.popleft()

    def close(self):
        self.closed = True
        self.rx.clear()

    def reset(self):
        self.actions.append('模拟复位')
        self.status = 0
        self.enabled = [False] * 6

    def enable(self):
        self.actions.append('模拟使能')
        self.enabled = [True] * 6


class ResumePreview(Panel):
    def __init__(self, handoff=False):
        self.bus = SimulatedBus(handoff)
        self.stage = 'idle'
        self.deadline = 0
        self.action_at = 0
        self.confirm_after = 0
        self.drive_stamp = None
        self.drives = ()
        self.notice = ''
        super().__init__(socket_factory=lambda: self.bus, log=lambda *_: None, auto_probe=False)
        self.setWindowTitle('离线演练 · PiPER 软件急停与恢复')
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
        self.stop.setText('模拟停止  STOP')
        self.layout().insertWidget(0, QtWidgets.QLabel('离线模拟总线 · 所有操作只改变模拟状态'))
        self.resume = QtWidgets.QPushButton('恢复遥操作')
        self.resume.setFocusPolicy(QtCore.Qt.NoFocus)
        self.resume.setAutoDefault(False)
        self.resume.clicked.connect(self.advance)
        self.layout().addWidget(self.resume)
        self.recovery = QtWidgets.QLabel()
        self.recovery.setWordWrap(True)
        self.layout().addWidget(self.recovery)
        self.link_result('offline-memory', '')
        self.notice = ''
        self.render()

    def press(self):
        self.cancel('')
        super().press()
        self.render()

    def cancel(self, reason):
        self.stage = 'idle'
        self.notice = reason

    def blocked(self):
        if not self.channel or self.bus.closed or not self.bus.online:
            return '模拟反馈中断'
        if self.drive_stamp is None or time.monotonic() - self.drive_stamp > 0.35:
            return '模拟反馈已过期'
        if self.bus.fault:
            return '模拟操作臂报告故障'
        if not self.bus.handoff:
            return '缺少主臂已交出控制权的确认。主臂静止或暂时没有报文，均不能确认交接。'
        if not self.bus.aligned:
            return '主臂目标与操作臂姿态未对齐'
        return ''

    def fresh_stop(self):
        c = self.channel
        return (c is not None and c.confirmed and c.last_status == 1 and c.last_seen is not None
                and time.monotonic() - c.last_seen <= 0.35)

    def advance(self):
        # Process current feedback again at the point of the operator's click.
        self.tick()
        if not self.resume.isEnabled():
            return
        if self.stage == 'idle' and self.fresh_stop():
            self.await_operator('confirm_reset')
        elif self.stage == 'confirm_reset' and self.fresh_stop():
            self.action_at = time.monotonic()
            self.deadline = self.action_at + 1
            self.stage = 'wait_reset'
            self.bus.reset()
        elif self.stage == 'confirm_enable':
            self.action_at = time.monotonic()
            self.deadline = self.action_at + 1
            self.stage = 'wait_enable'
            self.bus.enable()
        self.render()

    def await_operator(self, stage):
        self.stage = stage
        self.confirm_after = time.monotonic() + QtWidgets.QApplication.doubleClickInterval() / 1000

    def tick(self):
        if not self.channel:
            self.cancel('接口已断开')
            self.render()
            return
        self.drive_stamp, self.drives = self.bus.sample()
        super().tick()
        reason = self.blocked()
        if self.stage != 'idle' and reason:
            self.cancel(reason)
        elif self.stage in ('wait_reset', 'wait_enable'):
            if time.monotonic() >= self.deadline:
                self.cancel('反馈等待超时，请重新检查现场状态')
            elif self.channel.last_status not in (0, 1) or (self.stage == 'wait_enable' and self.channel.last_status == 1):
                self.cancel('操作臂报告异常或再次急停')
            elif (self.drive_stamp > self.action_at and self.channel.last_seen is not None
                  and self.channel.last_seen > self.action_at and self.channel.last_status == 0):
                if self.stage == 'wait_reset' and self.drives == (False,) * 6:
                    self.await_operator('confirm_enable')
                elif self.stage == 'wait_enable' and self.drives == (True,) * 6:
                    self.stage = 'complete'
        elif self.stage == 'confirm_reset' and not self.fresh_stop():
            self.cancel('已失去本次 STOP 确认')
        elif self.stage == 'confirm_enable' and (self.channel.last_status != 0 or self.drives != (False,) * 6):
            self.cancel('复位后的状态已改变')
        elif self.stage == 'complete' and (self.channel.last_status != 0 or self.drives != (True,) * 6):
            self.cancel('使能状态已改变')
        self.render()

    def render(self):
        stage = self.stage
        button, hint = {
            'idle': ('恢复遥操作', self.notice or '先按软件 STOP，等待操作臂反馈。'),
            'confirm_reset': ('已支撑操作臂，模拟复位', '复位可能失去保持力。确认操作臂已落稳、工作区已清空。'),
            'wait_reset': ('等待复位反馈', '等待新的正常状态及六轴失能反馈。'),
            'confirm_enable': ('现场确认，模拟重新使能', '模拟复位已确认。确认无人处于运动范围，再重新使能。'),
            'wait_enable': ('等待使能反馈', '等待新的正常状态及六轴使能反馈。'),
            'complete': ('模拟使能已确认', '模拟操作臂六轴已使能。实机跟随效果待验证。'),
        }[stage]
        problem = self.blocked()
        self.resume.setText(button)
        separate_click = stage not in ('confirm_reset', 'confirm_enable') or time.monotonic() >= self.confirm_after
        self.resume.setEnabled(separate_click and not problem and (
            (stage in ('idle', 'confirm_reset') and self.fresh_stop()) or stage == 'confirm_enable'))
        self.recovery.setText(problem or hint)

    def detach(self):
        super().detach()
        self.cancel('接口已断开')
        if hasattr(self, 'resume'):
            self.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulate-handoff', action='store_true',
                        help='模拟已确认交接控制权，用于演练后续步骤')
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    pane = ResumePreview(args.simulate_handoff)
    pane.show()
    app.exec_()
