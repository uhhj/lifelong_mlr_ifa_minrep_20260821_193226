"""Resume the two frozen pilot pairs sequentially, stopping on any failed stage."""
import os
import subprocess
import sys
import time
from .common import OUT, ROOT, read, write


def main():
    import fcntl
    lock = (OUT / 'pilot_pairs.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for source, target in ((6, 7), (7, 6)):
        unit = 's45_%d_%d' % (source, target)
        for stage in ('signals', 'target', 'evaluate_offline'):
            log = OUT / ('PILOT_%s_%s.log' % (stage, unit))
            started = time.time()
            with log.open('a') as output:
                run = subprocess.run([sys.executable, '-u', '-m', 'task_relation_forecast_v1.' + stage,
                                      '--seed', '45', '--source', str(source), '--target', str(target)],
                                     cwd=str(ROOT), env=environment, stdout=output, stderr=subprocess.STDOUT)
            write(OUT / 'pilot_stage_runs' / ('%s_%s.json' % (unit, stage)),
                  dict(returncode=run.returncode, started=started, ended=time.time(), log=str(log),
                       formal_training_resumed=False))
            if run.returncode:
                write(OUT / 'STATUS.json', dict(stage='PILOT_STAGE_FAILED', unit=unit,
                      failed_stage=stage, log=str(log), formal_training_resumed=False))
                raise RuntimeError('Pilot stage failed: %s %s' % (unit, stage))
    write(OUT / 'STATUS.json', dict(stage='PILOT_PAIRED_ADAPTATION_AND_OFFLINE_COMPLETE',
          source_updates_completed=2000, target_updates_completed=600, signals_sealed=2,
          rollouts_completed=0, remaining='Pilot native behavioral evaluation; full matrix not yet launched.',
          formal_training_resumed=False))


if __name__ == '__main__':
    main()
