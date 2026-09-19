"""Seed46 same-hardware 300-step diagnostic; never replaces failed acceptance."""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from .validate import ROOT, OLD, OUT, read, write, emit, sha

POINT = 's46_t8_e33_b148'
RUN = OUT / 'seed46_long_parity_diagnostic_v1'
STATES = Path('/root/sample_qp_seed46_long_parity_diagnostic_v1')
CHECKPOINT_STEPS = (20, 150, 174, 175, 200, 224, 225, 300)


def classify(same_hardware, legacy_history, optimized_history):
    if not same_hardware:
        return 'SAME_HARDWARE_IMPLEMENTATION_OR_EXECUTION_DIFFERENCE'
    if legacy_history and optimized_history:
        return 'SAME_HARDWARE_AND_HISTORICAL_EXACT'
    return 'SAME_HARDWARE_EXACT_HISTORICAL_ENVIRONMENT_DEPENDENCE'


class Guard:
    def __init__(self, started):
        self.started = started
        self.arm_started = started
        self.arm = 'setup'
        self.step = 0
        self.last = 0

    def check(self, force=False):
        import torch
        from functional_geometry_diagnostic_v1.runtime_guard import memory, size
        now = time.monotonic()
        if not force and now-self.last < 1:
            return
        self.last = now
        m = memory()
        assert now-self.started < 6*3600, 'Diagnostic six-hour total budget exceeded'
        assert now-self.arm_started < 4*3600, 'Diagnostic four-hour arm budget exceeded'
        assert m['rss'] < 28*2**30 and m['available'] > 2*2**30, 'Host memory budget'
        assert torch.cuda.memory_allocated() < 20*2**30, 'CUDA allocated budget'
        assert torch.cuda.memory_reserved() < 22*2**30, 'CUDA reserved budget'
        assert shutil.disk_usage(ROOT).free > 1*2**30, 'Data disk budget'
        assert shutil.disk_usage('/root').free > 4*2**30, 'Checkpoint disk reserve'
        assert size(STATES) < 8*2**30 and size(RUN) < 128*2**20, 'Output budget'
        write(RUN/'HEARTBEAT.json', dict(time=time.time(), pid=os.getpid(),
            arm=self.arm, step_in_progress=self.step, seconds=now-self.started,
            arm_seconds=now-self.arm_started, rss=m['rss'],
            cuda_allocated=torch.cuda.memory_allocated(),
            cuda_reserved=torch.cuda.memory_reserved()))


def run():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 3090'
    assert not RUN.exists() and not STATES.exists(), 'Preserve prior diagnostic; no overwrite'
    prior = OUT/'gpu_validation'
    assert read(prior/'STATUS.json')['status'] == 'FAILED_STOP'
    # Verify every previously frozen input, including the exact optimization code.
    files = dict(read(prior/'PROTOCOL_FROZEN.json')['files'])
    for path, expected in files.items():
        assert sha(path) == expected, ('Frozen input changed', path)
    for path in [prior/'STATUS.json', OUT/'LAST_ERROR.json',
                 prior/(POINT+'_PAIRED20.json'), prior/(POINT+'_events.jsonl'),
                 OLD/'TRAINING_MEMBERSHIP.json', OLD/'PROBE_LINEAGE.json',
                 Path(__file__)]:
        files[str(path.resolve())] = sha(path)
    RUN.mkdir(); STATES.mkdir()
    protocol = dict(point=POINT, hardware=torch.cuda.get_device_name(0),
        order=['legacy', 'optimized'], steps_per_arm=300,
        comparison='Exact full state component hashes after every step; historical endpoint separately',
        checkpoints=list(CHECKPOINT_STEPS), checkpoint_directory=str(STATES),
        scientific_changes=False, tolerance_changes=False, no_rollouts=True,
        no_seed47=True, no_acceptance_override=True,
        reference_windows=64, reference_refresh='every update',
        budgets=dict(total_hours=6, per_arm_hours=4, checkpoint_GiB=8,
                     host_rss_GiB=28, cuda_allocated_GiB=20, cuda_reserved_GiB=22),
        note='Same-hardware mismatch is diagnostic, not permission to relax exact acceptance. '
             'Environment dependence does not isolate GPU hardware as the sole cause.',
        git_sha=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        torch=torch.__version__, files=files)
    write(RUN/'PROTOCOL_FROZEN.json', protocol)
    write(RUN/'STATUS.json', dict(status='RUNNING', phase='restore', point=POINT,
                                 pid=os.getpid(), time=time.time()))
    from sample_step_closedloop_v1 import train as legacy
    from .adapter import update
    from .core import Workspace
    g = legacy.g
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started = time.monotonic(); guard = Guard(started)
    g.init_ddp(RUN/'ddp')
    try:
        spec, algo, ds, plan, base, Pold, lineage = legacy.assets(POINT)
        assert len(spec['batches']) == 300
        historical = read(OLD/'training'/POINT/'SAMPLE_STEP_QP'/'COMPLETE.json')
        paired20 = read(prior/(POINT+'_PAIRED20.json'))
        hashes = {}; summaries = {}; checkpoints = []
        for arm in protocol['order']:
            guard.arm = arm; guard.arm_started = time.monotonic(); guard.step = 0
            g.restore(algo, base)
            assert all(g.compare_states(base, g.capture(algo)).values())
            workspace = Workspace(g.params(algo)) if arm == 'optimized' else None
            hashes[arm] = []; update_seconds = []; first_mismatch = None
            for step in range(1, 301):
                guard.step = step; guard.check(force=True)
                torch.cuda.synchronize(); tick = time.perf_counter()
                if arm == 'legacy':
                    rec = legacy.update(algo, ds, plan, spec, Pold, lineage,
                                        POINT, 'SAMPLE_STEP_QP', step, guard)
                else:
                    rec = update(algo, ds, plan, spec, Pold, POINT, step, workspace, guard)
                torch.cuda.synchronize(); update_seconds.append(time.perf_counter()-tick)
                state = g.capture(algo)
                actual = {key:g.digest(value) for key,value in state.items()}
                hashes[arm].append(actual)
                comparison = None
                if arm == 'optimized':
                    comparison = {key:actual[key] == hashes['legacy'][step-1][key]
                                  for key in actual}
                    if not all(comparison.values()) and first_mismatch is None:
                        first_mismatch = step
                        write(RUN/'FIRST_SAME_HARDWARE_MISMATCH.json', dict(
                            step=step, fields=comparison, legacy=hashes['legacy'][step-1],
                            optimized=actual, time=time.time()))
                old_short = None
                if step <= 20:
                    old_key = 'legacy' if arm == 'legacy' else 'opt'
                    old_short = actual == paired20['state_hashes'][old_key][step-1]
                emit(RUN/(arm+'_state_hashes.jsonl'), dict(step=step, hashes=actual,
                    same_hardware_fields=comparison, previous_run_first20_exact=old_short))
                emit(RUN/(arm+'_events.jsonl'), rec)
                # Save evidence before any parity assertion, including failing final states.
                if step in CHECKPOINT_STEPS or step == first_mismatch:
                    path = STATES/(arm+'_step%03d.pth' % step)
                    torch.save(state, str(path)+'.tmp'); os.replace(str(path)+'.tmp', path)
                    checkpoints.append(dict(arm=arm, step=step, path=str(path),
                                            bytes=path.stat().st_size, sha256=sha(path)))
                    write(RUN/'CHECKPOINTS.json', checkpoints)
                if step == 1:
                    result = legacy.check_first(POINT, 'SAMPLE_STEP_QP', state)
                    write(RUN/(arm+'_FIRST_STEP.json'), dict(status='PASS', hashes=result))
                del state
                write(RUN/'STATUS.json', dict(status='RUNNING', arm=arm, point=POINT,
                    completed_steps=step, total_steps=300, pid=os.getpid(), time=time.time(),
                    elapsed_seconds=time.monotonic()-started,
                    arm_seconds=time.monotonic()-guard.arm_started,
                    update_seconds=sum(update_seconds), first_same_hardware_mismatch=first_mismatch))
                if step == 1 or step % 10 == 0:
                    print('DIAGNOSTIC_PROGRESS', arm, step, 'elapsed',
                          round(time.monotonic()-started, 1), flush=True)
            historical_fields = {k:v == historical['full_state_hashes'][k]
                                 for k,v in hashes[arm][-1].items()}
            summaries[arm] = dict(historical_exact=all(historical_fields.values()),
                historical_fields=historical_fields, endpoint_hashes=hashes[arm][-1],
                update_seconds=sum(update_seconds),
                wall_seconds=time.monotonic()-guard.arm_started,
                first_same_hardware_mismatch=first_mismatch)
            write(RUN/(arm+'_COMPLETE.json'), summaries[arm])
            del workspace; gc.collect(); torch.cuda.empty_cache()
        exact = hashes['legacy'] == hashes['optimized']
        for path, expected in files.items():
            assert sha(path) == expected, ('Frozen input changed during diagnostic', path)
        result = dict(status='DIAGNOSTIC_COMPLETE', same_hardware_all300_exact=exact,
            classification=classify(exact, summaries['legacy']['historical_exact'],
                                    summaries['optimized']['historical_exact']),
            arms=summaries, seconds=time.monotonic()-started,
            original_acceptance_status='FAILED_STOP_PRESERVED', seed47_started=False,
            limitation='One seed diagnostic. No performance acceptance or cross-hardware equivalence implied.')
        write(RUN/'COMPLETE.json', result)
        write(RUN/'STATUS.json', dict(status='DIAGNOSTIC_COMPLETE', time=time.time(),
            classification=result['classification'], same_hardware_all300_exact=exact))
        print('DIAGNOSTIC_COMPLETE', result['classification'], flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    # Lightweight tests for the diagnostic decision table, not scientific tolerances.
    assert classify(False, True, False) == 'SAME_HARDWARE_IMPLEMENTATION_OR_EXECUTION_DIFFERENCE'
    assert classify(True, True, True) == 'SAME_HARDWARE_AND_HISTORICAL_EXACT'
    assert classify(True, False, False) == 'SAME_HARDWARE_EXACT_HISTORICAL_ENVIRONMENT_DEPENDENCE'
    try:
        run()
    except BaseException:
        # A second invocation must not overwrite the first invocation's evidence.
        if RUN.exists() and (RUN/'STATUS.json').exists():
            status = read(RUN/'STATUS.json')
            if status.get('pid') == os.getpid():
                write(RUN/'ERROR.json', dict(time=time.time(), error=traceback.format_exc()))
                write(RUN/'STATUS.json', dict(status='FAILED_STOP', time=time.time(),
                    pid=os.getpid(), error_file=str(RUN/'ERROR.json')))
        raise
