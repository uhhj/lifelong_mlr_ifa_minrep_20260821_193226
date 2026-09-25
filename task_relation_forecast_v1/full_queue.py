"""Complete the frozen matrix after technical pilot validation; never gate on benefit."""
import os
import shutil
import subprocess
import sys
import time
from .common import ROOT, OUT, STORE, SEEDS, TASKS, read, write, sha


def audit_pilot():
    checks = {}
    checks['cpu_tests'] = read(OUT/'UNIT_TESTS.json')['status']=='PASS'
    for task in (6,7):
        directory=OUT/'sources'/('s45_t%d'%task)
        checks['source%d_complete'%task]=read(directory/'COMPLETE.json')['steps']==1000
        checks['source%d_native_parity'%task]=all(read(directory/'NATIVE_FIRST_STEP_PARITY.json')['checks'].values())
        checks['reference%d_prefix'%task]=read(directory/'COMPLETE.json')['reference300']==str(STORE/'sources'/('s45_t%d'%task)/'step0300.pt')
    for source,target in ((6,7),(7,6)):
        unit='s45_%d_%d'%(source,target);sig=OUT/'signals'/unit;seal=read(sig/'SEALED.json')
        checks[unit+'_seal']=all(sha(sig/name)==seal[key] for name,key in
              [('SIGNALS.json','signals_sha256'),('SHADOW_ROLLBACK.json','shadow_sha256'),('GN_GEOMETRY.json','geometry_sha256')])
        checks[unit+'_rollback']=all(read(sig/'SHADOW_ROLLBACK.json')['rollback'].values())
        checks[unit+'_first_step']=all(read(OUT/'transfers'/unit/'SHADOW_ACTUAL_FIRST_STEP_PARITY.json')['checks'].values())
        checks[unit+'_stream']=read(OUT/'transfers'/unit/'STREAM_PARITY.json')['status']=='PASS'
        checks[unit+'_target_complete']=read(OUT/'transfers'/unit/'COMPLETE.json')['steps']==300
        checks[unit+'_offline_complete']=read(OUT/'offline'/unit/'COMPLETE.json')['status']=='PASS'
        evidence=read(sig/'SIGNALS.json')
        checks[unit+'_data_firewall']=not evidence['E_eval_accessed'] and not evidence['source_A_accessed']
    manifest=read(OUT/'PILOT_BEHAVIOR_MANIFEST.json')
    checks['behavior_budget']=len(manifest['cells'])==12 and manifest['total_trials']==240
    for c in manifest['cells']:
        directory=OUT/'behavior'/('s%d'%c['seed'])/c['model_id']/('task%d'%c['task'])
        complete=read(directory/'COMPLETE.json');rounds=read(directory/'ROUNDS.json')
        pairing=read(OUT/'behavior/pairing'/('s%d_t%d.json'%(c['seed'],c['task'])))
        key=c['model_id']+'_t%d'%c['task']
        checks[key+'_files']=all(sha(directory/n)==h for n,h in complete['files'].items())
        checks[key+'_model']=complete['model_unchanged'] and sha(c['checkpoint'])==complete['checkpoint_sha256']
        checks[key+'_trials']=complete['trials']==20 and len(read(directory/'TRIALS.json'))==20
        checks[key+'_pairing']=pairing==dict(init=[r['init_sha256'] for r in rounds],
              initial_obs=[r['initial_obs_sha256'] for r in rounds],policy_rng_before=[r['policy_rng_before'] for r in rounds])
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                scope='Technical execution only; no success rate or transfer benefit is used as an advancement gate.',
                formal_training_resumed=False)
    write(OUT/'PILOT_TECHNICAL_AUDIT.json',result)
    assert all(checks.values()),result


def main():
    import fcntl
    lock=(OUT/'full_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Do not overlap this queue with either pilot queue.
    locks=[]
    for name in ('pilot_pairs.lock','behavior_queue.lock'):
        f=(OUT/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    audit_pilot()
    # Release the behavior lock so the native behavior child can acquire it later.
    fcntl.flock(locks[-1],fcntl.LOCK_UN)
    environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def execute(stage,seed=None,task=None,source=None,target=None):
        key='_'.join(str(v) for v in (stage,seed,task,source,target) if v is not None)
        if shutil.disk_usage(STORE).free<2*1024**3:raise RuntimeError('STORAGE_RESERVE_REACHED')
        command=[sys.executable,'-u','-m','task_relation_forecast_v1.'+stage]
        for k,v in [('seed',seed),('task',task),('source',source),('target',target)]:
            if v is not None:command += ['--'+k,str(v)]
        log=OUT/('FULL_'+key+'.log');started=time.time()
        write(OUT/'QUEUE_STATUS.json',dict(stage=stage,unit=key,started=started,status='RUNNING',formal_training_resumed=False))
        with log.open('a') as output:
            run=subprocess.run(command,cwd=str(ROOT),env=environment,stdout=output,stderr=subprocess.STDOUT)
        write(OUT/'full_stage_runs'/(key+'.json'),dict(command=command,started=started,ended=time.time(),
              returncode=run.returncode,log=str(log),formal_training_resumed=False))
        if run.returncode:
            write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',stage=stage,unit=key,log=str(log),formal_training_resumed=False))
            raise RuntimeError('Frozen full-matrix stage failed; '+str(log))
    for seed in SEEDS:
        for task in TASKS:
            if not (OUT/('CACHE_COMPLETE_s%d_t%d.json'%(seed,task))).exists():execute('cache',seed=seed,task=task)
            execute('source',seed=seed,task=task)
    pairs=[(seed,source,target) for seed in SEEDS for source in TASKS for target in TASKS if source!=target]
    for seed,source,target in pairs:execute('signals',seed=seed,source=source,target=target)
    signal_rows=[];seals={}
    for seed,source,target in pairs:
        unit='s%d_%d_%d'%(seed,source,target);directory=OUT/'signals'/unit;seal=read(directory/'SEALED.json')
        assert sha(directory/'SIGNALS.json')==seal['signals_sha256']
        signal_rows.append(read(directory/'SIGNALS.json')['scores']);seals[unit]=sha(directory/'SEALED.json')
    from .runtime import table
    table(OUT/'SIGNALS.csv',signal_rows)
    write(OUT/'SIGNALS_TABLE_SEALED.json',dict(rows=36,sha256=sha(OUT/'SIGNALS.csv'),unit_seals=seals,
          sealed_unix_time=time.time(),outcomes_used=False,formal_training_resumed=False))
    for seed,source,target in pairs:execute('target',seed=seed,source=source,target=target)
    for seed,source,target in pairs:execute('evaluate_offline',seed=seed,source=source,target=target)
    execute('behavior')
    write(OUT/'QUEUE_STATUS.json',dict(status='MEASUREMENTS_COMPLETE',source_models=12,pairs=36,
          rollouts=2880,next='Frozen prediction analysis and final report',formal_training_resumed=False))


if __name__=='__main__':main()
