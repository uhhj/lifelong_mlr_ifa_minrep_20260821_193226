"""Continue the original frozen seed46/47 plan without repeating seed45."""
import fcntl
import os
import subprocess
import time
import traceback
import zipfile
from .common import *
from .queue import stage


def pending_jobs(protocol):
    training = [(point, arm) for i, point in enumerate(protocol['points'])
                if i > 0 for arm in protocol['train_orders'][i]]
    evaluation = [(i, c) for i, c in enumerate(protocol['cells'])
                  if c['environment_seed'] in [46, 47]]
    assert len(training) == 6 and len(evaluation) == 18
    assert all(point.startswith(('s46_', 's47_')) for point, _ in training)
    return training, evaluation


def preserved(amendment):
    for name, expected in amendment['preserved_files'].items():
        assert sha(ROOT/name) == expected, ('Preserved seed45 asset changed', name)


def main():
    lock = (OUT/'remaining_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    p = read(OUT/'PROTOCOL_FROZEN.json')
    a = read(OUT/'REMAINING_SEEDS_AMENDMENT.json')
    assert sha(Path(__file__)) == a['implementation_sha256']
    assert read(OUT/'PREFLIGHT_COMPLETE.json')['status'] == 'PASS'
    assert read(OUT/'INDEPENDENT_AUDIT_SEED45.json')['status'] == 'PASS'
    assert read(OUT/'QUEUE_STATUS.json')['status'] == 'SEED45_COMPLETE_OTHERS_PAUSED'
    assert not (OUT/'REMAINING_SEEDS_STARTED.json').exists()
    preserved(a)
    training, evaluation = pending_jobs(p)
    for point, arm in training:
        assert not (OUT/'training'/point/arm).exists(), 'Partial branch requires explicit recovery'
    for i, c in evaluation:
        assert not (OUT/'evaluation'/c['point']/('task'+str(c['task']))/c['arm']).exists()
    os.environ['SAMPLE_QP_QUEUE_START'] = str(read(OUT/'QUEUE_STARTED.json')['time'])
    write(OUT/'REMAINING_SEEDS_STARTED.json', dict(
        time=time.time(), pid=os.getpid(), seeds=[46,47], train_jobs=6,
        evaluation_cells=18, additional_trials=900,
        implementation_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        original_protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'),
        amendment_sha256=sha(OUT/'REMAINING_SEEDS_AMENDMENT.json')))
    (OUT/'QUEUE.pid').write_text(str(os.getpid())+'\n')
    for point, arm in training:
        stage('train_'+point+'_'+arm,
              ['-m','sample_step_closedloop_v1.train',point,arm],
              p['resources']['train_job_hours']*3600)
    for i, c in evaluation:
        stage('eval_'+str(i).zfill(2),
              ['-m','sample_step_closedloop_v1.evaluate',str(i)],
              p['resources']['eval_cell_hours']*3600)
    stage('analysis',['-m','sample_step_closedloop_v1.analyze'],3600)
    stage('independent_audit',['-m','sample_step_closedloop_v1.audit'],3600)
    assert read(OUT/'INDEPENDENT_AUDIT.json')['status'] == 'PASS'
    preserved(a)
    write(OUT/'PRESERVATION_AUDIT.json',dict(status='PASS',files=len(a['preserved_files'])))
    note = ('\n\n## Execution scope history\n'
            'The user first paused seeds46/47 to complete seed45. After viewing the seed45 result, '
            'the user authorized these two originally specified seeds to continue. '
            'No method, sample sequence, endpoint, trial count or threshold was changed. '
            'The pooled result remains a staged development study, not an independent confirmation. '
            'The original seed45 report and raw results are preserved.\n')
    with (OUT/'FINAL_REPORT.md').open('a',encoding='utf-8') as f:
        f.write(note)
    write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',time=time.time(),
        seeds=[45,46,47],jobs=9,trials=1350,seed45_preserved=True,
        independent_audit='PASS'))
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='DELIVERY_MANIFEST.json']
    files += [ROOT/n for n in read(OUT/'FROZEN_INPUTS.json') if Path(n).suffix in ['.py','.yaml']]
    files += [Path(__file__)]
    write(OUT/'DELIVERY_MANIFEST.json',{f.relative_to(ROOT).as_posix():sha(f) for f in files})
    with zipfile.ZipFile('/root/sample_step_closedloop_v1_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[OUT/'DELIVERY_MANIFEST.json']:
            z.write(f,f.relative_to(ROOT).as_posix())


if __name__=='__main__':
    try:
        main()
    except BaseException:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',time=time.time(),
            pid=os.getpid(),error=traceback.format_exc()))
        raise
