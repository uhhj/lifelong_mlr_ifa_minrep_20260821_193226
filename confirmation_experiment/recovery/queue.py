"""Explicit recovery authorized by the user; never retries based on outcomes."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from confirmation_experiment.common import (ROOT,RUNS,REPORTS,read,write,sha,environment,revision)
from confirmation_experiment.recovery.common import (WORK,AUDIT,verify_original_code,verify_preserved_before_resume,verify_frozen_inputs,package_hashes)
from confirmation_experiment.recovery.commands import train,evaluate,rebuild_soft45
import confirmation_experiment.queue as original

def status(phase,**kw):
    x={'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**kw}
    write(WORK/'status.json',x);print(x,flush=True)

def start(cmd,folder,kind,env):
    folder.mkdir(parents=True,exist_ok=True)
    return original.start(cmd,folder,kind,env)

def evaluate45(arm,env,others=()):
    run=RUNS/'formal/seed45'/arm
    p=start(evaluate(45,arm,True),run,'eval_recovery',dict(env,CONFIRMATION_RUN_DIR=str(run),RECOVERY_REUSE_TASK6='1'))
    original.wait_done(p,others)
    assert read(run/'EVALUATION_COMPLETE.json')['current_process_native_calls']==9

def run():
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runs/replay_constraint_driver/queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (WORK/'STARTED.json').exists(),'Recovery already started; no automatic retry'
    verify_original_code();verify_preserved_before_resume();verify_frozen_inputs()
    assert read(AUDIT/'LAUNCHER_ISOLATION.json')['status']=='PASS'
    assert read(AUDIT/'EVAL_BOUNDARY_PARITY.json')['status']=='PASS'
    assert read(AUDIT/'PREFLIGHT.json')['status']=='PASS'
    release=read(AUDIT/'IMPLEMENTATION.json')
    assert release['package_sha256']==package_hashes()
    env=environment();env.update(CONFIRMATION_EVENT_LOGGING='1',CONFIRMATION_SNAPSHOTS='1',
         CONFIRMATION_BARRIERS='1',CONFIRMATION_IMPLEMENTATION_SHA=release['recovery_sha'])
    for seed in (46,47):assert not (RUNS/'formal'/f'seed{seed}').exists()
    for t,h in read(RUNS/'formal/seed45/mlr/TRAINING_COMPLETE.json')['checkpoint_sha256'].items():
        assert sha(RUNS/f'formal/seed45/mlr/run_001/mlrifa_{t}_model_ep99.pth')==h
    started={'pid':os.getpid(),'started_unix':time.time(),'recovery_sha':release['recovery_sha'],
             'original_implementation_sha':read(REPORTS/'IMPLEMENTATION.json')['implementation_sha'],
             'old_failure_preserved':str(RUNS/'FAILED.json')}
    write(WORK/'STARTED.json',started);write(RUNS/'RECOVERY_ACTIVE.json',started)
    status('reconstructing_soft45_task6',seed=45,arm='soft',task=6)
    rebuild=start(rebuild_soft45(),WORK,'rebuild',env)
    original.wait_event(AUDIT/'SOFT45_BOUNDARY.json',rebuild)
    verify_preserved_before_resume()
    status('remaining_evaluation',seed=45,arm='mlr',completed_cells=1)
    evaluate45('mlr',env,(rebuild,))
    status('remaining_training',seed=45,arm='soft',task=7)
    original.release(WORK/'RELEASE_SOFT45');original.wait_done(rebuild)
    original.validate_train(45,'soft',RUNS/'formal/seed45/soft')
    assert read(AUDIT/'SOFT45_TASK7_SAMPLING.json')['status']=='PASS'
    status('remaining_evaluation',seed=45,arm='soft',completed_cells=1)
    evaluate45('soft',env)
    write(REPORTS/'SEED45_COMPLETE.json',{'status':'PASS','seed':45,'both_arms_complete':True,'recovery_audit':str(AUDIT)})
    for seed in (46,47):
        a,b=[RUNS/'formal'/f'seed{seed}'/arm for arm in ('mlr','soft')]
        for arm,p in (('mlr',a),('soft',b)):
            p.mkdir(parents=True,exist_ok=False)
            write(p/'EXPERIMENT.json',{'seed':seed,'arm':arm,'original_implementation_sha':started['original_implementation_sha'],
                  'runtime_recovery_sha':release['recovery_sha'],'protocol_sha256':sha(ROOT/'confirmation_experiment/PROTOCOL.md'),
                  'train_command':train(seed,arm),'eval_command':evaluate(seed,arm)})
        status('task6_training',seed=seed,arm='mlr',task=6)
        pa=start(train(seed,'mlr'),a,'train',env);original.wait_event(a/'TASK6_TRAIN_READY.json',pa)
        status('task6_training',seed=seed,arm='soft',task=6)
        pb=start(train(seed,'soft'),b,'train',env);original.wait_event(b/'TASK6_TRAIN_READY.json',pb,(pa,))
        witness=original.check_task6(a,b)
        status('task6_evaluation_gate',seed=seed,arm='mlr',task=6)
        ea=start(evaluate(seed,'mlr'),a,'eval',dict(env,CONFIRMATION_RUN_DIR=str(a)))
        va=original.wait_event(a/'TASK6_EVAL_READY.json',ea,(pa,pb))
        status('task6_evaluation_gate',seed=seed,arm='soft',task=6)
        eb=start(evaluate(seed,'soft'),b,'eval',dict(env,CONFIRMATION_RUN_DIR=str(b)))
        vb=original.wait_event(b/'TASK6_EVAL_READY.json',eb,(pa,pb,ea));assert va==vb
        write(REPORTS/f'TASK6_PARITY_SEED{seed}.json',{'status':'PASS','train':witness,'eval':va})
        status('remaining_training',seed=seed,arm='mlr',task=7)
        original.release(a/'RELEASE_TRAIN');original.wait_done(pa,(pb,ea,eb));original.validate_train(seed,'mlr',a)
        status('full_evaluation',seed=seed,arm='mlr')
        original.release(a/'RELEASE_EVAL');original.wait_done(ea,(pb,eb))
        status('remaining_training',seed=seed,arm='soft',task=7)
        original.release(b/'RELEASE_TRAIN');original.wait_done(pb,(eb,));original.validate_train(seed,'soft',b)
        status('full_evaluation',seed=seed,arm='soft')
        original.release(b/'RELEASE_EVAL');original.wait_done(eb)
        write(REPORTS/f'SEED{seed}_COMPLETE.json',{'status':'PASS','seed':seed,'both_arms_complete':True})
    status('final_summary')
    verify_original_code()
    from confirmation_experiment.recovery.preservation import run as verify_preserved
    verify_preserved()
    subprocess.run([sys.executable,'-m','confirmation_experiment.summarize'],cwd=ROOT,env=env,check=True)
    results=read(REPORTS/'RESULTS.json');results['runtime_recovery']={
        'implementation':release,'boundary':read(AUDIT/'SOFT45_BOUNDARY.json'),
        'evaluation_boundary_parity':read(AUDIT/'EVAL_BOUNDARY_PARITY.json'),
        'preserved_results_files':read(AUDIT/'RESULT_PRESERVATION.json')['files'],
        'original_failure':read(RUNS/'FAILED.json'),'prefix_reconstruction_is_not_an_extra_scientific_arm':True}
    write(REPORTS/'RESULTS.json',results)
    with (REPORTS/'FINAL_REPORT.md').open('a') as f:f.write('''\n\n## Audited recovery of a launcher failure

The initial queue stopped after seed45 MLR training completed because overlapping PyTorch 1.13 standalone launchers shared a C10d store. Communication was isolated with separate ephemeral endpoints, verified with overlapping real NCCL processes and peer-exit survival. Scientific settings and original source files remained unchanged.

Seed45 soft task6 was replayed in a separate audit directory solely to reconstruct its lost live state. The original 15700-step loss trace, model, AdamW, scheduler, RNG and replay contents matched exactly before the same live process continued task7. Original completed checkpoints/results were retained; reconstruction outputs were not counted as an additional seed or arm.

Both completed seed45 task6 evaluations were reused. Model/RNG/config/backend equality at all remaining native evaluation boundaries was verified without robot trials. The interrupted, incomplete stage7 evaluation was replayed under its original protocol; failed-attempt logs remain archived. The primary matrix retains 20 trials per cell and no outcome-based reruns. Result preservation is audited in recovery_v1/RESULT_PRESERVATION.json.\n''')
    completion=read(REPORTS/'COMPLETION.json');completion['results_sha256']=sha(REPORTS/'RESULTS.json');completion['recovery_integrity']='PASS'
    write(REPORTS/'COMPLETION.json',completion)
    names=['FINAL_REPORT.md','RESULTS.json','primary_metrics.csv','paired_deltas.csv','per_task.csv','event_summary.json','event_summary.csv','COMPLETION.json']
    files=[REPORTS/n for n in names]+[REPORTS/f'TASK6_PARITY_SEED{s}.json' for s in (45,46,47)]
    files+=[AUDIT/n for n in ('SOFT45_BOUNDARY.json','SOFT45_TASK7_SAMPLING.json','RESULT_PRESERVATION.json')]
    subprocess.run(['git','add','-f','--']+[str(p.relative_to(ROOT)) for p in files],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Report confirmation results with audited launcher recovery'],cwd=ROOT,check=True)
    result_sha=revision();bundle=REPORTS/'RESULTS.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),release['recovery_sha']+'..HEAD','research/fixed-kappa-confirmation-v2'],cwd=ROOT,check=True)
    write(REPORTS/'PUBLICATION_READY.json',{'results_sha':result_sha,'bundle_sha256':sha(bundle),'branch':'research/fixed-kappa-confirmation-v2'})
    status('publication_pending',results_sha=result_sha)
    from confirmation_experiment.common import wait_for
    wait_for(REPORTS/'PUBLISHED.json',timeout=7*24*3600)
    assert read(REPORTS/'PUBLISHED.json')['results_sha']==result_sha
    done={'status':'PASS','results_commit':result_sha,'completed_unix':time.time(),'decision':results['decision']}
    write(WORK/'COMPLETE.json',done);write(RUNS/'COMPLETE.json',done);status('COMPLETE',decision=results['decision'])

if __name__=='__main__':
    try:run()
    except Exception as e:
        write(WORK/'FAILED.json',{'error':repr(e),'pid':os.getpid(),'time':time.time()});status('FAILED',error=repr(e))
        for p in original.children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        raise
