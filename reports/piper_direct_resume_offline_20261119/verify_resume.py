"""Headless tests through the actual Qt buttons, with network and CAN sockets forbidden."""
import os
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5 import QtWidgets
from resume_preview import ResumePreview


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.forbid_socket = patch('socket.socket', side_effect=AssertionError('physical transport used'))
        self.forbid_socket.start()
        self.addCleanup(self.forbid_socket.stop)
        self.clock = patch('time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.pane = ResumePreview(handoff=True)
        self.addCleanup(self.pane.close)
        self.bus = self.pane.bus
        self.pane.show()
        self.app.processEvents()

    def tick(self, seconds=.02):
        self.now.return_value += seconds
        self.pane.tick()

    def stop(self):
        self.pane.stop.click()
        self.tick()
        self.assertTrue(self.pane.channel.confirmed)

    def reset(self):
        self.stop()
        self.pane.resume.click()
        self.tick(1.)
        self.pane.resume.click()
        self.assertEqual(self.pane.stage, 'wait_reset')

    def test_direct_stop_feedback_does_not_transfer_control(self):
        self.bus.handoff = False
        self.assertEqual(self.bus.actions, [])
        self.stop()
        for _ in range(12):
            self.tick(.25)
            self.pane.resume.click()
        self.assertFalse(self.pane.resume.isEnabled())
        self.assertTrue(self.pane.stop.isEnabled())
        self.pane.close()
        self.assertEqual(self.bus.actions, ['STOP'])

    def test_reset_and_enable_need_separate_intent_and_fresh_ack(self):
        self.stop()
        self.pane.resume.click()
        self.pane.resume.click()
        self.assertEqual(self.bus.actions, ['STOP'])
        self.tick(1.)
        self.pane.resume.click()
        self.assertEqual(self.pane.stage, 'wait_reset')
        self.tick()
        self.assertEqual(self.pane.stage, 'confirm_enable')
        self.pane.resume.click()
        self.assertEqual(self.bus.actions, ['STOP', '模拟复位'])
        self.tick(1.)
        self.pane.resume.click()
        self.assertEqual(self.pane.stage, 'wait_enable')
        self.tick()
        self.assertEqual(self.pane.stage, 'complete')
        self.assertEqual(self.bus.actions, ['STOP', '模拟复位', '模拟使能'])
        self.pane.resume.click()
        self.assertEqual(len(self.bus.actions), 3)

    def test_stop_preempts_pending_enable(self):
        self.reset()
        self.tick()
        self.stop()
        self.tick(2.)
        self.assertEqual(self.pane.stage, 'idle')
        self.assertEqual(self.bus.actions, ['STOP', '模拟复位', 'STOP'])

    def test_partial_enable_times_out_without_retries(self):
        self.reset()
        self.tick()
        self.tick(1.)
        self.pane.resume.click()
        self.bus.enabled[-1] = False
        self.tick()
        self.assertEqual(self.pane.stage, 'wait_enable')
        self.tick(1.1)
        self.assertEqual(self.pane.stage, 'idle')
        self.assertIn('超时', self.pane.recovery.text())
        self.assertFalse(self.pane.resume.isEnabled())
        self.assertEqual(len(self.bus.actions), 3)

    def test_fault_or_lost_status_or_handoff_or_alignment_cancels(self):
        self.stop()
        for attr, bad, good in [('fault', True, False), ('online', False, True),
                                ('aligned', False, True), ('handoff', False, True)]:
            with self.subTest(attr=attr):
                self.pane.resume.click()
                self.assertEqual(self.pane.stage, 'confirm_reset')
                setattr(self.bus, attr, bad)
                self.tick(1.)
                self.pane.resume.click()
                self.assertEqual(self.pane.stage, 'idle')
                setattr(self.bus, attr, good)
                self.tick()
        self.assertEqual(self.bus.actions, ['STOP'])

    def test_external_stop_or_old_stop_does_not_authorize_resume(self):
        self.bus.status = 1
        self.tick()
        self.pane.resume.click()
        self.assertEqual(self.bus.actions, [])
        self.stop()
        self.bus.status = 0
        self.tick()
        self.bus.status = 1
        self.tick()
        self.assertFalse(self.pane.resume.isEnabled())

    def test_reset_without_full_feedback_never_progresses(self):
        self.reset()
        self.bus.enabled[2] = True
        self.tick()
        self.assertEqual(self.pane.stage, 'wait_reset')
        self.tick(1.1)
        self.assertEqual(self.pane.stage, 'idle')
        self.assertFalse(self.pane.resume.isEnabled())
        self.assertEqual(self.bus.actions, ['STOP', '模拟复位'])

    def test_stop_preempts_waiting_for_enable_ack(self):
        self.reset()
        self.tick()
        self.tick(1.)
        self.pane.resume.click()
        self.stop()
        self.tick(1.)
        self.assertEqual(self.pane.stage, 'idle')
        self.assertEqual(self.bus.actions[-1], 'STOP')
        self.assertTrue(self.pane.channel.confirmed)

    def test_link_loss_at_click(self):
        self.stop()
        self.pane.resume.click()
        self.tick(1.)
        self.bus.closed = True
        self.pane.resume.click()
        self.assertIsNone(self.pane.channel)
        self.assertFalse(self.pane.resume.isEnabled())
        self.assertEqual(self.bus.actions, ['STOP'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
