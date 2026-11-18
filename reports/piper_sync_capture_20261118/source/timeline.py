"""Read the verified follower transmit echoes and feedback without a CAN socket."""

from collections import defaultdict, deque
import json
import math
import struct
import time


JOINT_TX = (0x155, 0x156, 0x157)
JOINT_RX = (0x2A5, 0x2A6, 0x2A7)
READ_WINDOW_NS = 60_000_000


def decode_joints(frames):
    return [value * math.pi / 180000 for frame in frames for value in struct.unpack('>ii', frame)]


def last_at(history, timestamp):
    return next((row for row in reversed(history) if row['mono_ns'] <= timestamp), None)


def read_status(directory):
    path = directory/'status.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def following(status):
    age = time.time() - status.get('time', 0)
    return status.get('phase') == 'following' and 0 <= age < 1


class Timeline:
    def __init__(self):
        self.accepted = defaultdict(deque)
        self.pending_targets = {}
        self.pending_feedback = []
        self.feedback = deque(maxlen=2048)
        self.measured_grip = deque(maxlen=2048)
        self.targets = deque(maxlen=2048)
        self.target_grip = deque(maxlen=2048)
        self.mode = None
        self.watermark = 0

    def ingest(self, event):
        kind = event['kind']
        stamp = event['mono_ns']
        self.watermark = max(self.watermark, stamp)
        cid = event['can_id']
        data = bytes.fromhex(event['data_hex'])
        if kind == 'follower_target_tx_failed':
            raise ValueError('从臂目标发送失败')
        if kind == 'follower_target_tx_accepted':
            pending = self.accepted[cid]
            if len(pending) >= 128:
                raise ValueError('从臂目标缺少发送回显')
            pending.append(event)
        elif kind == 'can0_tx_echo':
            pending = self.accepted[cid]
            if not pending or pending[0]['data_hex'] != event['data_hex']:
                raise ValueError(f'从臂发送回显与下发记录不匹配 ({cid:#x})')
            accepted = pending.popleft()
            self.confirm_target(cid, data, stamp, accepted.get('source_mono_ns'))
            return {'id':cid, 'data_hex':event['data_hex'], 'host_monotonic_ns':stamp,
                    'source_mono_ns':accepted.get('source_mono_ns'), 'origin':'local_can_tx'}
        elif kind == 'follower_rx':
            self.read_feedback(cid, data, stamp)
        return None

    def confirm_target(self, cid, data, stamp, source):
        if cid == 0x151:
            if data[:2] != b'\x01\x01' or data[3] != 0xAD:
                raise ValueError('从臂控制模式与当前高跟随不符')
            self.mode = data.hex()
        elif cid == 0x159:
            self.target_grip.append(dict(mono_ns=stamp, width=struct.unpack('>i', data[:4])[0]/1000,
                                         frame=data.hex(), source_mono_ns=source))
        elif cid in JOINT_TX:
            group = self.pending_targets.setdefault(source, {})
            if cid in group or len(self.pending_targets) > 16:
                raise ValueError('从臂目标回显缺组或重复')
            group[cid] = (stamp, data)
            if len(group) == len(JOINT_TX):
                ordered = [group[i] for i in JOINT_TX]
                self.targets.append(dict(mono_ns=max(t for t,_ in ordered),
                                         joints=decode_joints([d for _,d in ordered]),
                                         frames=[d.hex() for _,d in ordered], source_mono_ns=source))
                del self.pending_targets[source]

    def read_feedback(self, cid, data, stamp):
        if cid == 0x2A8:
            self.measured_grip.append(dict(mono_ns=stamp, width=struct.unpack('>i', data[:4])[0]/1000))
        elif cid == JOINT_RX[0]:
            self.pending_feedback = [(stamp, data)]
        elif cid in JOINT_RX and self.pending_feedback:
            if cid != JOINT_RX[len(self.pending_feedback)] or not 0 <= stamp-self.pending_feedback[0][0] <= 25_000_000:
                self.pending_feedback = []
                return
            self.pending_feedback.append((stamp, data))
            if len(self.pending_feedback) == len(JOINT_RX):
                self.feedback.append(dict(mono_ns=stamp,
                    joints=decode_joints([d for _,d in self.pending_feedback]),
                    component_monotonic_ns=[t for t,_ in self.pending_feedback]))
                self.pending_feedback = []

    def sample_at(self, stamp):
        state = last_at(self.feedback, stamp)
        width = last_at(self.measured_grip, stamp)
        q = last_at(self.targets, stamp)
        g = last_at(self.target_grip, stamp)
        if not self.mode or any(item is None for item in (state,width,q,g)):
            return None
        measured = state['component_monotonic_ns']+[width['mono_ns']]
        if stamp-min(measured) > READ_WINDOW_NS:
            return None
        observation = dict(joints_rad=state['joints'], gripper_mm=width['width'],
                           sample_monotonic_ns=min(measured), component_monotonic_ns=measured)
        command = dict(joints_rad=q['joints'], gripper_mm=g['width'], origin='local_can_tx',
                       sample_monotonic_ns=max(q['mono_ns'],g['mono_ns']),
                       joint_sent_mono_ns=q['mono_ns'], gripper_sent_mono_ns=g['mono_ns'],
                       joint_source_mono_ns=q['source_mono_ns'], gripper_source_mono_ns=g['source_mono_ns'],
                       control_mode_hex=self.mode, frame_hex=q['frames']+[g['frame']])
        return observation, command


class TimelineTail:
    def __init__(self, directory):
        self.directory = directory
        self.stream = (directory/'can_timeline.jsonl').open('rb')
        self.timeline = Timeline()

    def read(self, limit=5000):
        for _ in range(limit):
            at = self.stream.tell()
            raw = self.stream.readline()
            if not raw.endswith(b'\n'):
                self.stream.seek(at)
                return
            event = json.loads(raw)
            yield event, self.timeline.ingest(event)

    def close(self):
        self.stream.close()
