"""Six preregistered runs with task6 barriers and the untouched native evaluator.

No performance-dependent branches before all six matrices finish. Exceptions
are integrity/execution failures only. No automatic retry of formal training.
"""
import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from confirmation_experiment.common import ROOT,RUNS,REPORTS,read,write,sha,environment,revision,state_digest
from confirmation_experiment.commands import train_command,eval_command
from confirmation_experiment.protocol import SEEDS
from confirmation_experiment.preflight import package_hashes

children=[]
def status(phase,**kw):
    obj={'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**kw}
    write(RUNS/'status.json',obj);print(obj,flush=True)
def start(cmd,run,kind,env):
    log=run/(kind+'_stdout.log')
    assert not log.exists(),str(log)
    with log.open('w') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p)
    write(run/(kind+'_process.json'),{'pid':p.pid,'command':cmd,'started_unix':time.time()})
    return p
def wait_event(path,process,others=(),timeout=24*3600):
    start=time.monotonic()
    while not path.exists():
        if process.poll() is not None:raise RuntimeError(f'Process {process.pid} exited {process.returncode} before {path}')
        for p in others:
            if p.poll() not in (None,0):raise RuntimeError(f'Companion process {p.pid} failed {p.returncode}')
        if time.monotonic()-start>timeout:raise TimeoutError(str(path))
        time.sleep(2)
    return read(path)
def wait_done(process,others=()):
    while process.poll() is None:
        for p in others:
            if p.poll() not in (None,0):raise RuntimeError(f'Companion process {p.pid} failed {p.returncode}')
        time.sleep(2)
    if process.returncode:raise RuntimeError(f'Process {process.pid} exited {process.returncode}')
def release(path):write(path,{'released_unix':time.time(),'reason':'preregistered integrity checks passed'})

def check_task6(mlr,soft):
    import torch
    a,b=[read(p/'TASK6_TRAIN_READY.json') for p in (mlr,soft)]
    assert a==b,('Task6 trace/state/RNG mismatch',a,b)
    parameter_hashes=[]
    for p in (mlr,soft):
        c=torch.load(p/'run_001/mlrifa_6_model_ep99.pth',map_location='cpu')
        digest=state_digest(c['state_dict']);parameter_hashes.append(digest)
        assert digest==a['model_state_sha256'],'ep99 differs from completed task6 model'
    assert parameter_hashes[0]==parameter_hashes[1]
    return dict(a,ep99_parameters_verified=True)

def validate_train(seed,arm,run):
    cfg=read(run/'run_001/config.json')
    baseline=read(ROOT/'runs/final_paper_text_faithful/mlr/seed42/run_001/config.json')
    for k in ('benchmark_name','policy','data','train','start_task','end_task','aug','past_data_rate',
              'ifa_enabled','repro_scheduler_mode','save_final_checkpoint','load_previous_model'):
        assert cfg[k]==baseline[k],('Formal config drift',seed,arm,k)
    assert cfg['seed']==seed
    for t in (6,7,8,9):assert (run/f'run_001/mlrifa_{t}_model_ep99.pth').is_file()
    with (run/'run_001/paper_repro_train_curve.csv').open() as f:curve=list(csv.DictReader(f))
    assert len(curve)==400 and {(int(r['stage']),int(r['epoch'])) for r in curve}=={(t,e) for t in (6,7,8,9) for e in range(100)}
    if arm=='soft':
        assert cfg['lifelong']['kappa']==.5 and cfg['lifelong']['projection_enabled'] is True
        expected=sum(int(r['batches']) for r in curve);count=0
        with (run/'events.jsonl').open() as f:
            for line in f:
                import json
                r=json.loads(line);count+=1
                assert r['seed']==seed and r['kappa']==.5 and r['global_optimizer_step']==count
        assert count==expected,('Incomplete event stream',count,expected)
    write(run/'TRAINING_COMPLETE.json',{'status':'PASS','epochs':400,'seed':seed,'arm':arm,
          'checkpoint_sha256':{str(t):sha(run/f'run_001/mlrifa_{t}_model_ep99.pth') for t in (6,7,8,9)}})

def run():
    RUNS.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'runs/replay_constraint_driver/queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUNS/'STARTED.json').exists(),'Formal queue already started; never silently rerun'
    pre=read(REPORTS/'PREFLIGHT_COMPLETE.json')
    assert pre['status']=='PASS' and read(REPORTS/'SEED_AUDIT.json')['status']=='PASS'
    for name,h in pre['historical_asset_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in pre['package_sha256'].items():assert sha(ROOT/name)==h,('Code changed after preflight',name)
    frozen=read(REPORTS/'FROZEN_INPUTS.json')
    assert sha(ROOT/'confirmation_experiment/PROTOCOL.md')==frozen['protocol_sha256']
    for name,h in frozen['code_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in frozen['input_sha256'].items():assert sha(ROOT/name)==h,name
    for seed,hashes in frozen['cache_sha256'].items():
        for name,h in hashes.items():assert sha(ROOT/name)==h,name
    implementation=read(REPORTS/'IMPLEMENTATION.json')
    assert implementation['package_sha256']==package_hashes(),'Uncommitted/unvalidated package change'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='research/fixed-kappa-confirmation-v2'
    env=environment();env.update(CONFIRMATION_EVENT_LOGGING='1',CONFIRMATION_SNAPSHOTS='1',CONFIRMATION_BARRIERS='1',
                                 CONFIRMATION_IMPLEMENTATION_SHA=implementation['implementation_sha'])
    write(RUNS/'STARTED.json',{'pid':os.getpid(),'started_unix':time.time(),'seeds':list(SEEDS),'arms':['mlr','soft'],
          'order':'seed45,46,47; within seed MLR task6, soft task6, task6 eval gate, MLR remainder+eval, soft remainder+eval',
          'starting_sha':frozen['starting_state']['starting_sha'],'implementation_sha':implementation['implementation_sha'],
          'launch_head':revision(),'preflight_sha256':sha(REPORTS/'PREFLIGHT_COMPLETE.json')})
    (RUNS/'queue.pid').write_text(str(os.getpid())+'\n')
    for seed in SEEDS:
        base=RUNS/'formal'/f'seed{seed}';a,b=base/'mlr',base/'soft'
        assert not a.exists() and not b.exists(),'Refusing existing formal output'
        for arm,p in (('mlr',a),('soft',b)):
            p.mkdir(parents=True)
            write(p/'EXPERIMENT.json',{'seed':seed,'arm':arm,'implementation_sha':implementation['implementation_sha'],
                  'protocol_sha256':frozen['protocol_sha256'],'frozen_inputs_sha256':sha(REPORTS/'FROZEN_INPUTS.json'),
                  'train_command':train_command(seed,arm),'eval_command':eval_command(seed,arm)})
        status('task6_training',seed=seed,arm='mlr',task=6)
        pa=start(train_command(seed,'mlr'),a,'train',env);wait_event(a/'TASK6_TRAIN_READY.json',pa)
        status('task6_training',seed=seed,arm='soft',task=6)
        pb=start(train_command(seed,'soft'),b,'train',env);wait_event(b/'TASK6_TRAIN_READY.json',pb,(pa,))
        witness=check_task6(a,b)
        status('task6_evaluation_gate',seed=seed,arm='mlr',task=6)
        ea=start(eval_command(seed,'mlr'),a,'eval',dict(env,CONFIRMATION_RUN_DIR=str(a)))
        va=wait_event(a/'TASK6_EVAL_READY.json',ea,(pa,pb))
        status('task6_evaluation_gate',seed=seed,arm='soft',task=6)
        eb=start(eval_command(seed,'soft'),b,'eval',dict(env,CONFIRMATION_RUN_DIR=str(b)))
        vb=wait_event(b/'TASK6_EVAL_READY.json',eb,(pa,pb,ea))
        assert va==vb,('task6 evaluation parity mismatch',va,vb)
        write(REPORTS/f'TASK6_PARITY_SEED{seed}.json',{'status':'PASS','train':witness,'eval':va})
        status('remaining_training',seed=seed,arm='mlr',task=7)
        release(a/'RELEASE_TRAIN');wait_done(pa,(pb,ea,eb));validate_train(seed,'mlr',a)
        status('full_evaluation',seed=seed,arm='mlr')
        release(a/'RELEASE_EVAL');wait_done(ea,(pb,eb))
        status('remaining_training',seed=seed,arm='soft',task=7)
        release(b/'RELEASE_TRAIN');wait_done(pb,(eb,));validate_train(seed,'soft',b)
        status('full_evaluation',seed=seed,arm='soft')
        release(b/'RELEASE_EVAL');wait_done(eb)
        for p in (a,b):assert read(p/'EVALUATION_COMPLETE.json')['native_evaluation_calls']==10
        write(REPORTS/f'SEED{seed}_COMPLETE.json',{'status':'PASS','seed':seed,'both_arms_complete':True})
    status('final_summary')
    subprocess.run([sys.executable,'-m','confirmation_experiment.summarize'],cwd=ROOT,env=env,check=True)
    # Report commit is scoped to this confirmation only, never git add -A.
    publish=['FINAL_REPORT.md','RESULTS.json','primary_metrics.csv','paired_deltas.csv','per_task.csv',
             'event_summary.json','event_summary.csv','COMPLETION.json']+[f'TASK6_PARITY_SEED{s}.json' for s in SEEDS]
    subprocess.run(['git','add']+[str((REPORTS/n).relative_to(ROOT)) for n in publish],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Report preregistered fixed-kappa confirmation on seeds 45-47'],cwd=ROOT,check=True)
    results_sha=revision()
    bundle=REPORTS/'RESULTS.bundle'
    assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),implementation['implementation_sha']+'..HEAD',
                    'research/fixed-kappa-confirmation-v2'],cwd=ROOT,check=True)
    write(REPORTS/'PUBLICATION_READY.json',{'results_sha':results_sha,'bundle_sha256':sha(bundle),
          'branch':'research/fixed-kappa-confirmation-v2'})
    status('publication_pending',results_sha=results_sha)
    from confirmation_experiment.common import wait_for
    wait_for(REPORTS/'PUBLISHED.json',timeout=7*24*3600)
    assert read(REPORTS/'PUBLISHED.json')['results_sha']==results_sha
    write(RUNS/'COMPLETE.json',{'status':'PASS','results_commit':revision(),'completed_unix':time.time(),
                              'decision':read(REPORTS/'RESULTS.json')['decision']})
    status('COMPLETE',decision=read(REPORTS/'RESULTS.json')['decision'])

if __name__=='__main__':
    try:run()
    except Exception as e:
        write(RUNS/'FAILED.json',{'error':repr(e),'pid':os.getpid(),'time':time.time(),
                                'reason':'execution/integrity failure, not a performance stopping rule'})
        status('FAILED',error=repr(e))
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        raise
