"""Feed the AgileX ROS 2 recorder without opening a CAN socket."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from ament_index_python.packages import get_package_prefix
from builtin_interfaces.msg import Time
from data_msgs.msg import Array, CaptureStatus
from data_msgs.srv import CaptureService
import rclpy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image

from cameras import CameraStream, configured_cameras
from timeline import TimelineTail, read_status


ROLES = ('overhead', 'wrist')
CAMERAS = dict(overhead='external', wrist='wrist')
NAMES = [f'joint_{i}' for i in range(1, 7)] + ['gripper_width_mm']
TASKS = {'B1': '夹起木块，放进左侧蓝碟', 'B2': '夹起木块，放进右侧浅色碟',
         'B3': '把木块推到左侧黄色区域', 'I1': '夹起木块，放进中间粉碟',
         'I2': '把木块推到右侧紫色区域', 'I3': '把木块放进中央红色精确放置区'}
PHASES = ('bound_waiting_input', 'following')
MAX_CAMERA_SKEW_NS = 20_000_000
LOG_SETTLE_NS = 150_000_000
OUTPUTS = ('camera/color/overhead', 'camera/color/wrist',
           'array/float32/follower_feedback', 'array/float32/follower_target')


def stamp(ns):
    seconds, fraction = divmod(ns, 1_000_000_000)
    return Time(sec=seconds, nanosec=fraction)


def is_bound(directory):
    status = read_status(directory)
    return (status.get('phase') in PHASES and 0 <= time.time() - status.get('time', 0) < 1
            and not (directory / 'STOP').exists() and not (directory / 'result.json').exists())


def persisted_samples(root):
    episode = root / 'episode0'
    keys = [{p.stem for p in (episode / stream).iterdir() if p.stem.replace('.', '').isdecimal()}
            for stream in OUTPUTS]
    if any(stream != keys[0] for stream in keys[1:]):
        raise RuntimeError('AgileX 写入的两路图像、从臂反馈和实际目标时间索引不一致')
    return len(keys[0])


class Live:
    def __init__(self, run, cleanup):
        self.run = run
        self.cameras = {}
        for role, cfg in configured_cameras().items():
            stream = CameraStream(cfg)
            cleanup.callback(stream.close)
            self.cameras[role] = stream
        self.tail = TimelineTail(run)
        cleanup.callback(self.tail.close)
        self.sequence = {role: 0 for role in ROLES}

    def samples(self, started, finished=None):
        if not is_bound(self.run):
            raise RuntimeError('遥操已停止或遥操日志已断流')
        for _ in self.tail.read(limit=16000):
            pass
        timeline = self.tail.timeline
        cutoff = min(time.monotonic_ns() - LOG_SETTLE_NS, timeline.watermark)
        if finished is not None:
            cutoff = min(cutoff, finished)
        recent = {role: iter([frame for frame in self.cameras[CAMERAS[role]].frames()
                             if frame['sequence'] > self.sequence[role]
                             and started <= frame['host_monotonic_ns'] <= cutoff]) for role in ROLES}
        pending = {role: next(recent[role], None) for role in ROLES}
        while all(pending.values()):
            times = {role: pending[role]['host_monotonic_ns'] for role in ROLES}
            aligned = max(times.values())
            if aligned - min(times.values()) <= MAX_CAMERA_SKEW_NS:
                yield aligned, dict(pending), timeline.observation_at(aligned), timeline.command_at(aligned)
                advance = ROLES
            else:
                advance = (min(times, key=times.get),)
            for role in advance:
                self.sequence[role] = pending[role]['sequence']
                pending[role] = next(recent[role], None)


class OfficialRecorder:
    def __init__(self, root, task, cleanup):
        self.root = root
        self.task = task
        self.node = rclpy.create_node('piper_capture_bridge')
        cleanup.callback(self.node.destroy_node)
        prefix = Path(get_package_prefix('data_tools'))
        executable = prefix / 'lib/data_tools/data_tools_dataCapture'
        log = cleanup.enter_context((root / 'data_tools.log').open('w'))
        self.process = subprocess.Popen([
            str(executable), '--ros-args', '--params-file', str(Path(__file__).with_name('piper_data_params.yaml')),
            '-p', 'useService:=true', '-p', 'episodeIndex:=0', '-p', 'hz:=1', '-p', 'timeout:=5',
            '-p', 'cropTime:=-1.0', '-p', f'datasetDir:={root}',
            '-p', 'instructions:=' + json.dumps(json.dumps([TASKS[task]], ensure_ascii=False)),
        ], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        cleanup.callback(self.close)
        self.client = self.node.create_client(CaptureService, '/data_tools_dataCapture/capture_service')
        self.images = {role: self.node.create_publisher(Image, f'/piper_record/{role}/image', 10) for role in ROLES}
        self.infos = {role: self.node.create_publisher(CameraInfo, f'/piper_record/{role}/camera_info', 10) for role in ROLES}
        self.states = {name: self.node.create_publisher(Array, f'/piper_record/follower_{name}', 10)
                       for name in ('feedback', 'target')}
        self.vendor_failed = False
        self.node.create_subscription(CaptureStatus, '/data_tools_dataCapture/status', self.status, 10)
        self.sidecar = cleanup.enter_context((root / 'alignment.jsonl').open('w', buffering=1))

    def status(self, result):
        self.vendor_failed |= result.fail

    def wait(self, condition):
        deadline = time.monotonic() + 8
        while not condition():
            if self.process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('AgileX 采集服务未就绪，查看 data_tools.log')
            rclpy.spin_once(self.node, timeout_sec=.05)

    def service(self, start):
        request = CaptureService.Request(start=start, end=not start, episode_index=0,
                                         dataset_dir=str(self.root), instructions=json.dumps([TASKS[self.task]]),
                                         task_name=self.task, task_descriptions=TASKS[self.task], task_id=self.task)
        response = self.client.call_async(request)
        self.wait(response.done)
        if not response.result().success:
            raise RuntimeError('AgileX 采集服务拒绝操作')

    def start(self):
        self.wait(self.client.service_is_ready)
        self.service(True)
        pubs = (*self.images.values(), *self.infos.values(), *self.states.values())
        self.wait(lambda: all(p.get_subscription_count() for p in pubs))
        return time.monotonic_ns()

    def publish(self, ns, pair, feedback, target):
        common = stamp(ns)
        for role, frame in pair.items():
            array = frame['image']
            height, width = array.shape[:2]
            info = CameraInfo(height=height, width=width)
            info.header.stamp = common
            info.header.frame_id = f'{role}_rgb'
            image = Image(height=height, width=width, encoding='bgr8', step=width * 3, data=array.tobytes())
            image.header = info.header
            self.infos[role].publish(info)
            self.images[role].publish(image)
        for name, value in (('feedback', feedback), ('target', target)):
            values = [] if value is None else [*value['joints_rad'], value['gripper_mm']]
            state = Array(shape=[len(values)], data=values, dim_description=[','.join(NAMES)])
            state.header.stamp = common
            self.states[name].publish(state)
        self.sidecar.write(json.dumps(dict(sample_monotonic_ns=ns, cameras={role: {
            'monotonic_ns': f['host_monotonic_ns'], 'sequence': f['sequence']} for role, f in pair.items()},
            feedback=feedback, target=target), separators=(',', ':')) + '\n')

    def stop(self):
        # Give the vendor subscriptions time to enqueue the last published sample.
        deadline = time.monotonic() + .3
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=.02)
        self.service(False)
        if self.vendor_failed:
            raise RuntimeError('AgileX 采集过程中报告数据流中断')

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    bindings = args.project / 'official_ui_teleop_v1/binding_runs'
    for task, description in TASKS.items():
        print(task, description)
    task = input('任务编号 > ').strip().upper()
    if task not in TASKS:
        parser.error('任务编号无效')
    input('现场完成遥操绑定并保持静止。按回车开始录制 > ')
    runs = [run for run in bindings.iterdir() if run.is_dir() and is_bound(run)]
    if len(runs) != 1:
        raise RuntimeError('需要且只允许一个已绑定的遥操会话')
    destination = args.project / 'agilex_data_tools/recordings' / f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    destination.mkdir(parents=True, mode=0o700)
    result = dict(task=task, description=TASKS[task], source_run=runs[0].name,
                  outcome='aborted', pairs=0, valid_targets=0, valid_feedback=0)
    print(f'保存到 {destination}', flush=True)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        with ExitStack() as cleanup:
            live = Live(runs[0], cleanup)
            recorder = OfficialRecorder(destination, task, cleanup)
            started = recorder.start()
            print('录制已开始。按回车结束本条示教。', flush=True)
            last_pair = time.monotonic()

            def publish(sample):
                recorder.publish(*sample)
                result['pairs'] += 1
                result['valid_feedback'] += sample[2] is not None
                result['valid_targets'] += sample[3] is not None

            try:
                while not select.select([sys.stdin], [], [], 0)[0]:
                    for sample in live.samples(started):
                        publish(sample)
                        last_pair = time.monotonic()
                    rclpy.spin_once(recorder.node, timeout_sec=.01)
                    if recorder.vendor_failed or recorder.process.poll() is not None or time.monotonic() - last_pair > 3:
                        raise RuntimeError('采集未收到连续同步帧，检查两个相机和遥操日志')
                if sys.stdin.readline() == '':
                    raise RuntimeError('操作终端已关闭')
                finished = time.monotonic_ns()
                while time.monotonic_ns() - finished <= 2 * LOG_SETTLE_NS:
                    for sample in live.samples(started, finished):
                        publish(sample)
                    rclpy.spin_once(recorder.node, timeout_sec=.02)
            finally:
                recorder.stop()
            if not result['pairs'] or persisted_samples(destination) != result['pairs']:
                raise RuntimeError('AgileX 写入数量与同步采样数量不一致')
            outcome = input('示教结果 [s 成功 / f 失败 / a 中止] > ').strip().lower()
            result['outcome'] = {'s': 'success', 'f': 'failed'}.get(outcome, 'aborted')
    finally:
        result['episode'] = 'episode0'
        temporary = destination / 'episode_summary.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(destination / 'episode_summary.json')
        rclpy.shutdown()


if __name__ == '__main__':
    main()
