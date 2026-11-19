"""Offscreen Qt tests on a memory transport. No physical CAN socket is opened."""
import os
import socket
import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.append(str(Path(__file__).resolve().parents[1] / 'piper_direct_stop_20261119' / 'source'))
from PyQt5 import QtWidgets
from stop_panel import FRAME, STAMP, STATUS, TIMESPEC, Panel, StopChannel

STOP_FRAME = bytes.fromhex('50010000080000000100000000000000')
RESET_FRAME = bytes.fromhex('50010000080000000200000000000000')


class Wire:
    def __init__(self):
        self.tx = []
        self.rx = deque()
        self.reject = False
        self.closed = False

    def send(self, packet):
        assert not self.closed
        if self.reject:
            raise OSError('offline send failure')
        self.tx.append(packet)
        return len(packet)

    def recvmsg(self, *_):
        if not self.rx:
            raise BlockingIOError()
        return self.rx.popleft()

    def close(self):
        self.closed = True


class ResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.now = 9_000_000_000
        for name, fn in [('time.time_ns', lambda: self.now), ('time.monotonic', lambda: self.now / 1e9),
                         ('socket.socket', unittest.mock.Mock(side_effect=AssertionError('real socket')))]:
            p = patch(name, new=fn)
            p.start()
            self.addCleanup(p.stop)
        self.wire = Wire()
        self.events = []
        self.p = Panel(lambda: self.wire, lambda event, detail: self.events.append(event), auto_probe=False)
        self.addCleanup(self.p.close)
        self.p.show()
        self.p.link_result(7, '')
        self.now += 1_100_000_000

    def frame(self, code=1, cid=STATUS, flags=0, error=0, stamp=None):
        stamp = self.now if stamp is None else stamp
        anc = [(socket.SOL_SOCKET, STAMP, TIMESPEC.pack(*divmod(stamp, 10**9)))]
        data = bytes([0, code, 0, 0, 0, 0]) + error.to_bytes(2, 'big')
        self.wire.rx.append((FRAME.pack(cid, 8, data), anc, flags, None))

    def prepare(self):
        self.frame()
        self.p.tick()
        self.assertTrue(self.p.reset.isEnabled())
        self.p.reset.click()
        self.assertFalse(self.p.apply_reset.isEnabled())
        self.p.attest.setChecked(True)

    def submit(self):
        self.prepare()
        self.p.apply_reset.click()
        self.assertEqual(self.wire.tx, [RESET_FRAME])

    def test_external_stop_reset_and_new_normal_feedback(self):
        self.submit()
        self.p.apply_reset.click()
        self.frame(code=0, stamp=self.now-1)
        self.now += 1_000_000
        self.frame(code=0, flags=socket.MSG_DONTROUTE)
        self.frame(code=0, cid=0x2a2)
        self.frame(code=1)
        self.p.tick()
        self.assertNotIn('arm_reset_confirmed', self.events)
        self.frame(code=0)
        self.p.tick()
        self.assertIn('arm_reset_confirmed', self.events)
        self.assertEqual(self.wire.tx, [RESET_FRAME])

    def test_stop_preempts_before_and_after_submission(self):
        self.prepare()
        self.p.stop.click()
        self.p.apply_reset.click()
        self.assertEqual(self.wire.tx, [STOP_FRAME])
        self.now += 1_000_000
        self.prepare()
        self.p.apply_reset.click()
        self.p.stop.click()
        self.now += 1_000_000
        self.frame(code=0)
        self.p.tick()
        self.assertEqual(self.wire.tx, [STOP_FRAME, RESET_FRAME, STOP_FRAME])
        self.assertNotIn('arm_reset_confirmed', self.events)

    def test_stop_retries_do_not_overlap_reset(self):
        self.p.stop.click()
        self.frame(stamp=self.now-1)
        self.p.tick()
        self.assertFalse(self.p.reset.isEnabled())
        self.now += 110_000_000
        self.p.tick()
        self.assertEqual(self.wire.tx, [STOP_FRAME, STOP_FRAME])

    def test_activity_queued_at_click_revokes_consent(self):
        for flags in (0, socket.MSG_DONTROUTE):
            self.p.link_result(7, '')
            self.prepare()
            self.frame(cid=0x155, flags=flags)
            self.p.apply_reset.click()
            self.assertEqual(self.wire.tx, [])
            self.now += 1_100_000_000

    def test_fault_and_temp_exit_cancel_consent_even_in_one_batch(self):
        for code, error in ((1, 0x100), (0, 0)):
            self.prepare()
            self.frame(code=code, error=error)
            self.frame(code=1)
            self.p.apply_reset.click()
            self.assertEqual(self.wire.tx, [])

    def test_no_stale_consent_on_loss_or_reconnect(self):
        self.prepare()
        self.now += 300_000_000
        self.p.apply_reset.click()
        self.assertEqual(self.wire.tx, [])
        self.p.link_result(None, '硬件只听')
        self.p.reset.click()
        self.assertFalse(self.p.reset.isEnabled())
        self.assertFalse(self.p.stop.isEnabled())
        self.assertEqual(self.wire.tx, [])

    def test_no_transmit_after_cancel_expiry_or_old_adapter_check(self):
        self.prepare()
        self.p.dismiss_reset()
        self.p.finish_reset()
        self.prepare()
        self.now += 31_000_000_000
        self.p.link_result(7, '')
        self.frame()
        self.p.finish_reset()
        self.prepare()
        self.now += 2_000_000_000
        self.frame()
        self.p.tick()
        self.p.finish_reset()
        self.assertEqual(self.wire.tx, [])

    def test_backlog_at_final_click_blocks_reset(self):
        self.prepare()
        for _ in range(128):
            self.frame()
        self.frame(cid=0x151)
        self.p.apply_reset.click()
        self.assertEqual(self.wire.tx, [])

    def test_missing_ack_never_retries_or_confirms_late(self):
        self.submit()
        self.now += 1_100_000_000
        self.frame(code=0)
        self.p.tick()
        self.assertNotIn('arm_reset_confirmed', self.events)
        self.assertIn('未知', self.p.reset_status.text())
        self.assertEqual(self.wire.tx, [RESET_FRAME])

    def test_failure_is_not_success(self):
        self.prepare()
        self.wire.reject = True
        self.p.apply_reset.click()
        self.frame(code=0)
        self.p.tick()
        self.assertIn('reset_send_failed', self.events)
        self.assertNotIn('arm_reset_confirmed', self.events)
        self.assertEqual(self.wire.tx, [])

    def test_error_after_submission(self):
        self.submit()
        self.now += 1_000_000
        self.frame(code=0, error=1)
        self.p.tick()
        self.assertNotIn('arm_reset_confirmed', self.events)
        self.assertIn('异常', self.p.reset_status.text())


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux kernel timestamps')
class KernelTest(unittest.TestCase):
    def test_reset_round_trip(self):
        receiver, peer = socket.socketpair(type=socket.SOCK_DGRAM)
        with receiver, peer:
            receiver.setblocking(False)
            receiver.setsockopt(socket.SOL_SOCKET, STAMP, 1)
            peer.settimeout(1)
            events = []
            ch = StopChannel(receiver, lambda event, detail: events.append(event))
            ch.control_seen -= 2
            peer.send(FRAME.pack(STATUS, 8, bytes([0, 1, 0, 0, 0, 0, 0, 0])))
            ch.poll()
            self.assertTrue(ch.submit_reset(ch.confirm_generation))
            self.assertEqual(peer.recv(FRAME.size), RESET_FRAME)
            peer.send(FRAME.pack(STATUS, 8, bytes(8)))
            ch.poll()
            self.assertEqual(events, ['reset_submitted', 'arm_reset_confirmed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
