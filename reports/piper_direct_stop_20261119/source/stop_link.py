"""Identify the onsite direct CAN adapter without changing its configuration."""
from pathlib import Path
import json
import subprocess

PORT = 'can0'
BINDING = Path(__file__).with_name('can0_stop_binding.json')


def usb_identity(port=PORT):
    device = (Path('/sys/class/net') / port / 'device').resolve(strict=True)
    for parent in (device, *device.parents):
        if (parent/'idVendor').is_file() and (parent/'serial').is_file():
            return {key: (parent/key).read_text().strip() for key in ('idVendor','idProduct','serial')}
    raise RuntimeError('无法识别旁听用 USB-CAN')


def snapshot():
    result = subprocess.run(['/usr/sbin/ip','-j','-d','link','show','dev',PORT],
                            check=True,capture_output=True,text=True,timeout=1)
    link, = json.loads(result.stdout)
    info = link['linkinfo']['info_data']
    if (link['linkinfo']['info_kind'] != 'can' or
            info.get('bittiming_const',{}).get('name') != 'gs_usb' or
            not {'UP','LOWER_UP'}.issubset(link['flags']) or
            info.get('state') not in ('ERROR-ACTIVE','ERROR-WARNING','ERROR-PASSIVE') or
            info.get('bittiming',{}).get('bitrate') != 1_000_000):
        raise RuntimeError('can0 未就绪、已 bus-off 或波特率不符')
    modes = set(info.get('ctrlmode',[]))
    if modes.intersection({'LOOPBACK','PRESUME-ACK','ONE-SHOT'}):
        raise RuntimeError('can0 工作模式不符')
    identity = dict(port=PORT, ifindex=link['ifindex'], usb=usb_identity(),
                    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    return identity, 'LISTEN-ONLY' in modes


def check_path(*, allow_listen_only):
    identity, silent = snapshot()
    if silent:
        if allow_listen_only:
            return identity
        raise RuntimeError('can0 硬件只听，STOP 无法发送')
    if not BINDING.is_file() or identity != json.loads(BINDING.read_text()):
        raise RuntimeError('当前可发送接口与本轮确认的直连接口不符')
    return identity
