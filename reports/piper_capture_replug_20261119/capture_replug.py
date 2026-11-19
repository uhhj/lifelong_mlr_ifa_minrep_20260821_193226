"""Preflight for direct capture only. Active episodes only check the link."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile

from direct_can import check_capture_link
from stop_link import BINDING, PORT, check_path, snapshot, usb_identity

NET = Path('/sys/class/net')
BOOT = Path('/proc/sys/kernel/random/boot_id')
IP = '/usr/sbin/ip'
SUDO = '/usr/bin/sudo'


def atomic_json(path, identity):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(identity, f)
            f.flush()
            os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_link(expected):
    links = json.loads(subprocess.check_output([IP, '-d', '-j', 'link', 'show'], text=True, timeout=5))
    numeric = 'if' + str(expected['ifindex'])
    if any(numeric in (x['ifname'], *x.get('altnames', ())) for x in links):
        raise RuntimeError('接口名称与内核索引别名冲突')
    link = next((x for x in links if x['ifname'] == PORT), {})
    detail = link.get('linkinfo', {})
    info = detail.get('info_data', {})
    if (link.get('ifindex') != expected['ifindex'] or detail.get('info_kind') != 'can' or
            info.get('bittiming_const', {}).get('name') != 'gs_usb' or
            usb_identity(PORT) != expected['usb'] or BOOT.read_text().strip() != expected['boot']):
        raise RuntimeError('检查期间 USB-CAN 身份发生变化')
    if set(info.get('ctrlmode', ())) & {'LOOPBACK', 'PRESUME-ACK', 'ONE-SHOT', 'FD'}:
        raise RuntimeError('can0 工作模式不符')
    return set(link['flags']), info


def check_down(identity):
    flags, info = read_link(identity)
    if 'UP' in flags or info.get('state') != 'STOPPED':
        raise RuntimeError('can0 已被其他程序启用或存在故障，保留现场状态')
    return info


def restore_down(identity, pending):
    # sudo reads from the operator's terminal only if the existing ticket has expired.
    subprocess.run([SUDO, '-v'], check=True)
    check_down(identity)
    atomic_json(pending, identity)
    # iproute2's if<index> fallback pins the request to this kernel interface instance.
    # Replugging another device as "can0" cannot retarget either command to the new index.
    base = [SUDO, '-n', '--', IP, 'link', 'set', 'dev', 'if' + str(identity['ifindex'])]
    subprocess.run(base + ['type', 'can', 'bitrate', '1000000', 'listen-only', 'off'], check=True, timeout=5)
    info = check_down(identity)
    if info.get('bittiming', {}).get('bitrate') != 1_000_000 or 'LISTEN-ONLY' in info.get('ctrlmode', ()):
        raise RuntimeError('can0 位时序或工作模式尚未配置成功')
    subprocess.run(base + ['up'], check=True, timeout=5)


def prepare_capture_link(port):
    if port != PORT:
        raise RuntimeError('录制接口与现场绑定不一致')
    with (BINDING.parent/'direct_install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            check_capture_link(port)
            return
        except (OSError, RuntimeError, subprocess.CalledProcessError) as original:
            device = NET/port
            if not (device/'ifindex').is_file():
                raise RuntimeError('未检测到 can0，请检查 USB-CAN 连接') from original
            old = json.loads(BINDING.read_text())
            index = int((device/'ifindex').read_text())
            now = dict(port=port, boot=BOOT.read_text().strip(), usb=usb_identity(port), ifindex=index)
            if now['usb'] != old['usb'] or old['boot'] != now['boot'] or old['port'] != port:
                raise RuntimeError('当前 USB-CAN 与原绑定不符，请核对设备') from original
            if index == old['ifindex']:
                raise RuntimeError(f'can0 接口实例未变化，未自动重启：{original}') from original
        pending = BINDING.with_name('capture_replug_pending.json')
        flags, info = read_link(now)
        if 'UP' not in flags and info.get('state') == 'STOPPED':
            print('检测到原 USB-CAN 重插，正在恢复接口。若系统提示，请现场输入登录密码。', flush=True)
            restore_down(now, pending)
        elif not pending.is_file() or json.loads(pending.read_text()) != now:
            raise RuntimeError('新接口并非本入口正在恢复的 DOWN 实例，保留原绑定')
        ident, silent = snapshot()
        if silent or ident != now:
            raise RuntimeError('恢复期间接口发生变化，保留原绑定')
        atomic_json(BINDING, ident)
        check_path(allow_listen_only=False)
        pending.unlink(missing_ok=True)
        print(f'已恢复同一 USB-CAN：{port}，1 Mbit/s 可收发，接口编号 {index}。', flush=True)
