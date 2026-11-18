"""Exercise AgileX persistence with archived CAN and synthetic RGB, without hardware."""
import argparse
from contextlib import ExitStack
from decimal import Decimal
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions

from capture import OfficialRecorder
from timeline import Timeline


def replay(path):
    timeline = Timeline()
    idle = Timeline()
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            timeline.ingest(event)
            if event['kind'] == 'follower_rx':
                idle.ingest(event)
            aligned = timeline.sample_at(timeline.watermark)
            if aligned:
                assert idle.observation_at(timeline.watermark) == aligned[0]
                assert idle.command_at(timeline.watermark) is None
                return timeline.watermark, *aligned
    raise AssertionError('Archive contains no aligned command and feedback')


def verify(destination, idle_samples, sample_count):
    episode = destination / 'episode0'
    timestamps = [json.loads(line) for line in (destination / 'alignment.jsonl').read_text().splitlines()]
    assert len(timestamps) == sample_count
    inputs = {
        'overhead': episode / 'camera/color/overhead',
        'wrist': episode / 'camera/color/wrist',
        'feedback': episode / 'array/float32/follower_feedback',
        'target': episode / 'array/float32/follower_target',
    }
    for name, path in inputs.items():
        assert len([p for p in path.iterdir() if p.stem.replace('.', '').isdecimal()]) == sample_count, (name, path)
    for seq, row in enumerate(timestamps):
        key = f'{Decimal(row["sample_monotonic_ns"]) / Decimal(1_000_000_000):.6f}'
        for role, intensity in (('overhead', 21), ('wrist', 161)):
            image_file, = [p for p in inputs[role].glob(key + '.*')]
            assert cv2.imread(str(image_file))[0, 0, 0] == intensity
        for name in ('target', 'feedback'):
            values = np.load(inputs[name] / (key + '.npy'), allow_pickle=False)['data'].ravel().tolist()
            expected = row[name]
            if expected is None:
                assert name == 'target' and seq < idle_samples and values == []
            else:
                assert np.allclose(values, [*expected['joints_rad'], expected['gripper_mm']], rtol=1e-7, atol=1e-9)
    command_times = {row['target']['sample_monotonic_ns'] for row in timestamps if row['target'] is not None}
    assert len(command_times) == 1
    assert all(row['target'] is None for row in timestamps[:idle_samples])
    assert all(row['target'] is not None for row in timestamps[idle_samples:])
    print(f'OFFICIAL_SAVED pairs={sample_count} feedback={sample_count} idle_missing_target={idle_samples} '
          f'confirmed_and_held={sample_count-idle_samples} timestamps_match=true', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timeline', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    historical, observation, command = replay(args.timeline)
    args.destination.mkdir(mode=0o700)
    sample_count = 75
    idle_samples = 15
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        with ExitStack() as cleanup:
            recorder = OfficialRecorder(args.destination, 'B1', cleanup)
            start = recorder.start()
            imgs = {role: np.full((480, 640, 3), intensity, dtype=np.uint8)
                    for role, intensity in (('overhead', 21), ('wrist', 161))}
            for seq in range(sample_count):
                now = start + seq * 33_333_000
                pair = {role: dict(image=image, sequence=seq+1,
                                   host_monotonic_ns=now-(2_000_000 if role == 'wrist' else 0))
                        for role, image in imgs.items()}
                state = dict(observation, component_monotonic_ns=[
                    now - (historical - t) for t in observation['component_monotonic_ns']],
                             sample_monotonic_ns=now - (historical - observation['sample_monotonic_ns']))
                target = None
                if seq >= idle_samples:
                    shift = start + idle_samples * 33_333_000 - historical
                    target = {key: (value + shift if key.endswith('_mono_ns') and value is not None else value)
                              for key, value in command.items()}
                    target['sample_monotonic_ns'] += shift
                    assert target['sample_monotonic_ns'] <= now
                recorder.publish(now, pair, state, target)
                rclpy.spin_once(recorder.node, timeout_sec=0)
                time.sleep(max(0, (now - time.monotonic_ns()) / 1_000_000_000 + .033))
            recorder.stop()
        verify(args.destination, idle_samples, sample_count)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
