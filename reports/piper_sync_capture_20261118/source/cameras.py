"""Independent RGB camera capture on the remote host."""
from collections import deque
import fcntl
import os
from pathlib import Path
import struct
import threading
import time
import cv2

def supports_rgb(node):
    fd = os.open(node, os.O_RDONLY | os.O_NONBLOCK)
    try:
        for index in range(32):
            request = bytearray(64)
            struct.pack_into('II', request, 0, index, 1)
            try:
                fcntl.ioctl(fd, 0xc0405602, request, True)
            except OSError:
                return False
            if bytes(request[44:48]) in (b'YUYV',b'MJPG',b'RGB3',b'BGR3'):
                return True
        return False
    finally:
        os.close(fd)


def configured_cameras():
    found = {'external': [], 'wrist': []}
    for node in Path('/sys/class/video4linux').glob('video*'):
        device = (node/'device').resolve()
        usb = next((p for p in (device,*device.parents) if (p/'product').is_file()), None)
        if usb is None:
            continue
        product = (usb/'product').read_text().strip()
        lower = product.lower()
        if 'realsense' not in lower:
            continue
        role = 'external' if '435i' in lower else 'wrist' if '405' in lower else None
        path = '/dev/'+node.name
        if role and supports_rgb(path):
            serial = (usb/'serial').read_text().strip()
            found[role].append(dict(kind='v4l2', device=path, hardware_id=serial, model=product))
    if any(len(options) != 1 for options in found.values()):
        raise ValueError('需要各识别到一个 D435i 俯视彩色通道和 D405 腕部彩色通道：'
                         +str({role:len(items) for role,items in found.items()}))
    return {role:items[0] for role,items in found.items()}

class CameraStream:
    def __init__(self,config):
        self.config=dict(config);self.frame=None;self.history=deque(maxlen=60);self.error='waiting';self.stop=threading.Event();self.lock=threading.Lock();self.sequence=0
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        cap=None
        try:
            cap=cv2.VideoCapture(self.config['device'],cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_FPS,30);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if not cap.isOpened():raise RuntimeError('Camera cannot open')
            while not self.stop.is_set():
                started=time.monotonic_ns()
                ok,image=cap.read()
                if not ok:raise RuntimeError('Camera read failed')
                now=time.monotonic_ns();self.sequence+=1
                value={'image':image,'sequence':self.sequence,'host_monotonic_ns':now,'host_wall_ns':time.time_ns(),'read_started_monotonic_ns':started,
                  'hardware_timestamp_ms':None,'timestamp_domain':'host_delivery'}
                with self.lock:self.frame=value;self.history.append(value);self.error=None
        except Exception as e:self.error=str(e)
        finally:
            if cap:cap.release()
    def latest(self):
        with self.lock:return dict(self.frame) if self.frame else None
    def frames(self):
        with self.lock:return tuple(self.history)
    def close(self):self.stop.set();self.thread.join(timeout=1.5)
