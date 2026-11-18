"""Check the clock mapping and native grouping against archived local timelines."""
import json
from pathlib import Path
import sys

from core import Abort, EXPECTED, JIDS, NativeInput
from transport_base import CanClock, NS_PER_SECOND


def replay(path):
    with path.open() as stream:
        frames = [row for line in stream if (row := json.loads(line))['kind'] == 'master_rx'
                  and row['can_id'] in JIDS]
    first = frames[0]
    clock = CanClock(first['kernel_time_ns'] - first['mono_ns'])
    check = NativeInput(EXPECTED)
    old_reversals = groups = 0
    previous = first
    for frame in frames:
        old_reversals += frame['mono_ns'] < previous['mono_ns']
        previous = frame
        stamp = clock.project(frame['kernel_time_ns'], frame['observed_mono_ns'])
        last_group = check.last_complete
        check.put(frame['can_id'], bytes.fromhex(frame['data_hex']), stamp / NS_PER_SECOND)
        groups += last_group != check.last_complete
    print(f'{path.parent.name}: frames={len(frames)} old_reversals={old_reversals} fixed_reversals=0 groups={groups}')
    assert groups > 0
    return old_reversals


def test_clock_step():
    clock = CanClock(100 * NS_PER_SECOND)
    observed = NS_PER_SECOND
    current_wall = observed + clock.realtime_offset_ns
    assert clock.age(current_wall, current_wall, observed) == 0
    for step in (-25_000_000, 25_000_000):
        try:
            clock.age(current_wall + step, current_wall + step, observed)
        except Abort as exc:
            assert 'CAN clock discontinuity' in str(exc)
        else:
            raise AssertionError('wall clock step accepted')


if __name__ == '__main__':
    test_clock_step()
    assert sum(replay(Path(filename)) for filename in sys.argv[1:]) >= 1
