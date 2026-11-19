"""Exercise the recorder entry with temporary files and simulated link administration."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import capture
import capture_replug as app


class Preflight(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.binding = self.root/'binding.json'
        self.device = self.root/'net/can0'
        self.device.mkdir(parents=True)
        (self.device/'ifindex').write_text('11')
        self.boot = self.root/'boot'
        self.boot.write_text('test-boot')
        self.usb = dict(idVendor='test', idProduct='test', serial='test-adapter')
        self.old = dict(port='can0', ifindex=10, boot='test-boot', usb=self.usb)
        self.new = dict(self.old, ifindex=11)
        self.binding.write_text(json.dumps(self.old))
        for k,v in dict(BINDING=self.binding,BOOT=self.boot,NET=self.device.parent).items():
            stack.enter_context(patch.object(app,k,v))
        self.check = stack.enter_context(patch.object(app,'check_capture_link',side_effect=RuntimeError('down')))
        self.strict = stack.enter_context(patch.object(app,'check_path'))
        self.snap = stack.enter_context(patch.object(app,'snapshot',return_value=(self.new,False)))
        self.hardware = stack.enter_context(patch.object(app,'usb_identity',return_value=self.usb))
        self.run_cmd = stack.enter_context(patch.object(subprocess,'run',side_effect=self.simulate_command))
        self.ip_json = stack.enter_context(patch.object(subprocess,'check_output',side_effect=self.details))
        self.index=11
        self.up=False
        self.state='STOPPED'
        self.rate=0
        self.other_links=[]
        self.calls=[]

    def simulate_command(self, args, **kwargs):
        self.calls.append(args)
        if args==[app.SUDO,'-v']:
            return
        base=[app.SUDO,'-n','--',app.IP,'link','set','dev','if11']
        self.assertEqual(args[:len(base)],base)
        if args[len(base):]==['type','can','bitrate','1000000','listen-only','off']:
            self.rate=1_000_000
        else:
            self.assertEqual(args[len(base):],['up'])
            self.up=True

    def details(self,*a,**kw):
        return json.dumps([dict(ifname='can0',ifindex=self.index,flags=['UP'] if self.up else [],
            linkinfo=dict(info_kind='can',info_data=dict(state=self.state,bittiming=dict(bitrate=self.rate),
                bittiming_const=dict(name='gs_usb')))),*self.other_links])

    def test_replug(self):
        app.prepare_capture_link('can0')
        self.assertEqual(json.loads(self.binding.read_text()),self.new)
        self.assertEqual(len(self.calls),3)
        self.strict.assert_called_once_with(allow_listen_only=False)

    def test_healthy(self):
        self.check.side_effect=None
        app.prepare_capture_link('can0')
        self.assertFalse(self.calls)

    def test_same_instance_down(self):
        (self.device/'ifindex').write_text('10')
        with self.assertRaises(RuntimeError):app.prepare_capture_link('can0')
        self.assertFalse(self.calls)

    def test_device_missing(self):
        (self.device/'ifindex').unlink()
        with self.assertRaisesRegex(RuntimeError,'未检测到'):app.prepare_capture_link('can0')
        self.assertFalse(self.calls)

    def test_different_serial_or_boot(self):
        self.hardware.return_value=dict(self.usb,serial='other')
        with self.assertRaises(RuntimeError):app.prepare_capture_link('can0')
        self.hardware.return_value=self.usb
        self.boot.write_text('other-boot')
        with self.assertRaises(RuntimeError):app.prepare_capture_link('can0')
        self.assertFalse(self.calls)

    def test_bus_off(self):
        self.state='BUS-OFF'
        self.up=True
        with self.assertRaises(RuntimeError):app.prepare_capture_link('can0')
        self.assertFalse(self.calls)

    def test_numeric_name_collision(self):
        self.other_links=[dict(ifname='other',altnames=['if11'])]
        with self.assertRaises(RuntimeError):app.prepare_capture_link('can0')
        self.assertFalse(self.calls)

    def test_replaced_while_authenticating(self):
        self.run_cmd.side_effect=lambda *args,**kwargs:setattr(self,'index',12)
        with self.assertRaises(RuntimeError):app.prepare_capture_link('can0')
        self.assertEqual(self.run_cmd.call_count,1)
        self.assertEqual(json.loads(self.binding.read_text()),self.old)

    def test_password_declined(self):
        self.run_cmd.side_effect=subprocess.CalledProcessError(1,['sudo','-v'])
        with self.assertRaises(subprocess.CalledProcessError):app.prepare_capture_link('can0')
        self.assertEqual(json.loads(self.binding.read_text()),self.old)

    def test_retry_after_up_before_binding_write(self):
        with patch.object(app,'snapshot',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):app.prepare_capture_link('can0')
        self.assertEqual(json.loads(self.binding.read_text()),self.old)
        total=len(self.calls)
        app.prepare_capture_link('can0')
        self.assertEqual(len(self.calls),total)
        self.assertEqual(json.loads(self.binding.read_text()),self.new)

    def test_actual_capture_main_routes(self):
        class ReachedEpisodeDirectory(Exception):
            pass
        runs=self.root/'official_ui_teleop_v1/binding_runs'
        runs.mkdir(parents=True)
        for direct in (True,False):
            if not direct:(runs/'relay').mkdir()
            def intercept(path,*args,**kwargs):
                self.assertEqual(path.parent,self.root/'agilex_data_tools/recordings')
                raise ReachedEpisodeDirectory
            with (patch.object(sys,'argv',['capture.py','--project',str(self.root)]),
                  patch('builtins.input',side_effect=['B1','']),
                  patch.object(capture,'is_bound',return_value=True),
                  patch.object(capture,'prepare_capture_link',wraps=app.prepare_capture_link) as gate,
                  patch.object(Path,'mkdir',intercept),self.assertRaises(ReachedEpisodeDirectory)):
                capture.main()
            self.assertEqual(gate.call_count,int(direct))


if __name__=='__main__':
    unittest.main()
