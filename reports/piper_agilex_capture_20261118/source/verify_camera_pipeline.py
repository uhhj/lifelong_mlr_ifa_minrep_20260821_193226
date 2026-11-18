"""Exercise real cameras and the production pairing loop with replayed CAN observations."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import rclpy
from rclpy.signals import SignalHandlerOptions

from capture import Live, OfficialRecorder, persisted_samples
from verify_capture import replay


class ReplayTail:
    def __init__(self, original_ns, state, command):
        self.original_ns = original_ns
        self.started = time.monotonic_ns()
        self.state = state
        self.command = command
        self.timeline = self

    def read(self, limit):
        return iter(())

    def close(self):
        pass

    @property
    def watermark(self):
        return time.monotonic_ns()

    def observation_at(self, ns):
        shift = ns - self.original_ns
        return dict(self.state, sample_monotonic_ns=self.state['sample_monotonic_ns'] + shift,
                    component_monotonic_ns=[t + shift for t in self.state['component_monotonic_ns']])

    def command_at(self, ns):
        command_start = self.started + 3_000_000_000
        if ns < command_start:
            return None
        shift = command_start - self.original_ns
        data = {key: (value + shift if key.endswith('_mono_ns') and value is not None else value)
                for key, value in self.command.items()}
        return dict(data, sample_monotonic_ns=self.command['sample_monotonic_ns'] + shift)


def main(archive, output):
    fixture = ReplayTail(*replay(archive))
    output.mkdir(mode=0o700)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        with ExitStack() as cleanup, patch('capture.TimelineTail', return_value=fixture), patch('capture.is_bound', return_value=True):
            live = Live(output, cleanup)
            recorder = OfficialRecorder(output, 'B1', cleanup)
            start = recorder.start()
            end = time.monotonic() + 5
            count = 0
            while time.monotonic() < end:
                for sample in live.samples(start):
                    assert all(f['image'].shape == (480, 640, 3) for f in sample[1].values())
                    recorder.publish(*sample)
                    count += 1
                rclpy.spin_once(recorder.node, timeout_sec=.01)
            recorder.stop()
        rows = [json.loads(line) for line in (output / 'alignment.jsonl').read_text().splitlines()]
        skew = [abs(row['cameras']['overhead']['monotonic_ns'] - row['cameras']['wrist']['monotonic_ns']) for row in rows]
        missing = sum(row['target'] is None for row in rows)
        assert count == persisted_samples(output) == len(rows)
        assert count >= 100 and missing >= 10 and count - missing >= 10
        assert all(row['feedback'] is not None for row in rows)
        print(f'CAMERA_DRY_RUN pairs={count} idle_missing_target={missing} '
              f'feedback={count} max_pair_skew_ms={max(skew)/1e6:.3f} four_timelines_match=true', flush=True)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
