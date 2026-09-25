"""Unique, paired native rollout cells; shared references are evaluated once."""
import argparse
import os
import subprocess
import sys
import time
import shutil
from .common import ROOT, OUT, read, write, freeze, sha, stable_seed
from .runtime import model_path
# Required in spawned environment workers before historical LIBERO unpickling.
from cb_ifa_experiment.gpu_runtime import bind
bind()


def cells(pilot=False):
    seeds = (45,) if pilot else (45, 46, 47)
    tasks = (6, 7) if pilot else (6, 7, 8, 9)
    result = []
    for seed in seeds:
        def add(kind, task, source=None, target=None):
            if kind == 'base':
                model_id = 'base';path = model_path(seed, kind)
            elif kind == 'source':
                model_id = 'source%d' % source;path = model_path(seed, kind, source=source, step=1000)
            elif kind == 'reference':
                model_id = 'reference%d' % source;path = model_path(seed, kind, source=source, step=300)
            else:
                model_id = 'transfer%d_%d' % (source, target)
                path = model_path(seed, kind, source=source, target=target, step=300)
            result.append(dict(seed=seed, model_id=model_id, kind=kind, source=source, target=target,
                          task=task, checkpoint=str(path), environment_seed=seed,
                          round_policy_seeds=[stable_seed('task_relation_forecast_v1', 'rollout', seed, task, r)
                                              for r in range(2)]))
        for task in tasks:add('base', task)
        for source in tasks:
            for task in tasks:add('source', task, source=source)
        for task in tasks:add('reference', task, source=task)
        for source in tasks:
            for target in tasks:
                if source != target:
                    for task in (source, target):add('transfer', task, source, target)
    assert len(result) == (12 if pilot else 144)
    return result


def directory(cell):
    return OUT / 'behavior' / ('s%d' % cell['seed']) / cell['model_id'] / ('task%d' % cell['task'])


def accept_paired_retry(cell, attempt):
    source=OUT/'behavior_attempts'/('s%d'%cell['seed'])/cell['model_id']/('task%d'%cell['task'])/('attempt'+attempt)
    target=directory(cell);complete=read(source/'COMPLETE.json')
    assert complete['cell']==cell and complete['model_unchanged'] and complete['status']=='PASS'
    assert all(read(source/'PAIRING_CHECK.json')['checks'].values())
    assert sha(cell['checkpoint'])==complete['checkpoint_sha256']
    assert all(sha(source/n)==h for n,h in complete['files'].items())
    a,b=read(target/'CONFIG.json'),read(source/'CONFIG.json')
    differences=[k for k in set(a)|set(b) if a.get(k)!=b.get(k)]
    assert set(differences)<=set(['epoch_dir','experiment_dir','experiment_root','test_log_file'])
    assert not (target/'COMPLETE.json').exists()
    previous={}
    for name in ('ROUNDS.json','TRIALS.json','PAIRING_CHECK.json'):
        if (target/name).exists():
            archived=target/('FAILED_ATTEMPT000_'+name);assert not archived.exists()
            previous[name]=sha(target/name);os.replace(target/name,archived)
        shutil.copy2(source/name,target/name)
    retry_config='RETRY'+attempt+'_CONFIG.json';assert not (target/retry_config).exists()
    shutil.copy2(source/'CONFIG.json',target/retry_config)
    record=dict(accepted_attempt=str(source),accepted_attempt_complete_sha256=sha(source/'COMPLETE.json'),
          accepted_unix_time=time.time(),archived_original_files=previous,config_differences=differences,
          selection='Single fixed retry; acceptance uses exact initialization/RNG/model checks only, never success count',
          failure_root_cause='UNRESOLVED_INTERMITTENT_OBSERVATION_MISMATCH',formal_training_resumed=False)
    write(target/'RETRY_ACCEPTANCE.json',record);complete['accepted_retry']=record
    complete['files']={n:sha(target/n) for n in ['ROUNDS.json','TRIALS.json','PAIRING_CHECK.json','CONFIG.json',retry_config,'RETRY_ACCEPTANCE.json']}
    write(target/'COMPLETE.json',complete)


def run_cell(cell, attempt=None):
    from pathlib import Path
    from cb_ifa_experiment.gpu_runtime import torch, seed_all, digest, eval_config
    import robomimic.utils.obs_utils as ObsUtils
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import get_task_embs
    from libero.lifelong import metric
    from paper_qp_rollout_v1.evaluate import Observer
    dest = directory(cell)
    if attempt is not None:
        assert attempt.isdigit(), 'Diagnostic attempt must be an explicit numeric ID'
        dest = OUT/'behavior_attempts'/('s%d'%cell['seed'])/cell['model_id']/('task%d'%cell['task'])/('attempt'+attempt)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'COMPLETE.json').exists():
        complete = read(dest / 'COMPLETE.json')
        assert sha(cell['checkpoint']) == complete['checkpoint_sha256']
        for name, digest_value in complete['files'].items():assert sha(dest/name)==digest_value
        return
    assert not (dest / 'STARTED.json').exists(), 'Interrupted rollout requires an explicit replay audit'
    assert Path(cell['checkpoint']).exists()
    write(dest / 'STARTED.json', dict(cell=cell, pid=os.getpid(), started=time.time(),
          diagnostic_attempt=attempt, formal_training_resumed=False))
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
    cfg = eval_config(cell['seed'], dest)
    assert cfg.eval.n_eval == 20
    write(dest / 'CONFIG.json', cfg)
    seed_all(cell['seed'])
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': dict(cfg.data.obs.modality)})
    bench = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    bench.set_task_embs(get_task_embs(cfg, [bench.get_task(i).language for i in range(bench.n_tasks)]))
    algo = get_algo_class(cfg.lifelong.algo)(bench.n_tasks, cfg).to('cuda:0')
    state = torch.load(cell['checkpoint'], map_location='cpu')['model']
    expected = {n for n in algo.policy.state_dict() if n.startswith(('temporal_transformer.', 'policy_head.'))}
    assert set(state) == expected, ('Checkpoint parameter carrier mismatch', set(state)^expected)
    frozen = digest({k:v.cpu() for k,v in algo.policy.state_dict().items() if k not in expected})
    loaded = algo.policy.load_state_dict(state, strict=False)
    assert not loaded.unexpected_keys and not (set(loaded.missing_keys)&expected)
    assert frozen == digest({k:v.cpu() for k,v in algo.policy.state_dict().items() if k not in expected})
    assert digest(state) == digest({k:v.cpu() for k,v in algo.policy.state_dict().items() if k in expected})
    algo.eval();torch.set_grad_enabled(False);before = digest(algo.policy.state_dict())
    factory = metric.SubprocVectorEnv;instances=[]
    class EarlyPairingObserver(Observer):
        def set_init_state(self,states):
            obs=super().set_init_state(states)
            shared=OUT/'behavior'/'pairing'/('s%d_t%d.json'%(cell['seed'],cell['task']))
            if shared.exists():
                import numpy as np
                expected=read(shared);index=len(self.rounds)
                actual=dict(init=self.active['init_sha256'],initial_obs=self.active['initial_obs_sha256'],
                            policy_rng_before=self.active['policy_rng_before'])
                checks={k:actual[k]==expected[k][index] for k in actual}
                write(dest/('INITIAL_PAIRING_round%d.json'%index),dict(checks=checks,actual=actual,
                      expected={k:expected[k][index] for k in actual},formal_training_resumed=False))
                if not all(checks.values()):
                    arrays={};paths={}
                    def visit(v,path):
                        if isinstance(v,np.ndarray) and v.dtype.hasobject:
                            for i,item in enumerate(v.flat):visit(item,path+'/'+str(i))
                        elif isinstance(v,dict):
                            for k in sorted(v):visit(v[k],path+'/'+k)
                        elif isinstance(v,(list,tuple)):
                            for i,item in enumerate(v):visit(item,path+'/'+str(i))
                        else:
                            key='a%04d'%len(arrays);arrays[key]=np.asarray(v).copy();paths[key]=path
                    visit(obs,'obs')
                    tmp=dest/'MISMATCH_OBSERVATIONS.tmp.npz';np.savez_compressed(tmp,**arrays)
                    os.replace(tmp,dest/'MISMATCH_OBSERVATIONS.npz')
                    write(dest/'FAILURE_CONTEXT.json',dict(reason='Rollout initial pairing mismatch',
                          cell=cell,round=index,checks=checks,actual=actual,
                          expected={k:expected[k][index] for k in actual},observation_array_paths=paths,
                          before_policy_actions_this_round=True,formal_training_resumed=False))
                    write(dest/'PARTIAL_ROUNDS.json',self.rounds)
                    self.inner.close()
                    raise AssertionError('Rollout initial pairing mismatch before policy actions: '+str(checks))
            return obs
    def wrapped(*args, **kwargs):
        obj = EarlyPairingObserver(factory(*args, **kwargs), cell, dest, False);instances.append(obj);return obj
    metric.SubprocVectorEnv = wrapped;begin = time.monotonic()
    try:
        values = metric.evaluate_success(cfg, algo, bench, [cell['task']], None, see=cell['seed'])
    finally:
        metric.SubprocVectorEnv = factory
    rounds = [r for observer in instances for r in observer.rounds]
    assert len(instances)==1 and len(rounds)==2
    trials = [dict(seed=cell['seed'],model_id=cell['model_id'],task=cell['task'],
                   trial=r['round']*10+i,success=int(v),first_success_step=r['first'][i],policy_steps=r['count'])
              for r in rounds for i,v in enumerate(r['success'])]
    assert len(trials)==20 and abs(float(values[0])-sum(t['success'] for t in trials)/20)<1e-12
    assert before == digest(algo.policy.state_dict())
    task = bench.get_task(cell['task'])
    init = Path(cfg.init_states_folder)/task.problem_folder/task.init_states_file
    assert len(torch.load(init))>=20
    # The same seed/task must have identical initialization and round RNG across all models.
    pair_evidence = dict(init=[r['init_sha256'] for r in rounds],
                         initial_obs=[r['initial_obs_sha256'] for r in rounds],
                         policy_rng_before=[r['policy_rng_before'] for r in rounds])
    shared = OUT/'behavior'/'pairing'/('s%d_t%d.json'%(cell['seed'],cell['task']))
    # Persist the actual completed rounds before any pairing assertion. Diagnostics
    # must never discard the first mismatching fields together with the result.
    write(dest/'ROUNDS.json',rounds);write(dest/'TRIALS.json',trials)
    expected_pair=read(shared) if shared.exists() else pair_evidence
    pairing_checks={key:expected_pair[key]==pair_evidence[key] for key in pair_evidence}
    write(dest/'PAIRING_CHECK.json',dict(status='PASS' if all(pairing_checks.values()) else 'FAIL',
          checks=pairing_checks,expected=expected_pair,actual=pair_evidence,
          source_pairing_sha256=sha(shared) if shared.exists() else None,
          diagnostic_attempt=attempt,formal_training_resumed=False))
    if not all(pairing_checks.values()):
        write(dest/'FAILURE_CONTEXT.json',dict(reason='Rollout initial pairing mismatch',
              cell=cell,checks=pairing_checks,checkpoint_sha256=sha(cell['checkpoint']),
              native_evaluator_sha256=sha(ROOT/'libero/lifelong/metric.py'),
              completed_rollouts_retained=20,formal_training_resumed=False))
        raise AssertionError('Rollout initial pairing mismatch: '+str(pairing_checks))
    if not shared.exists():freeze(shared,pair_evidence)
    write(dest/'COMPLETE.json',dict(status='PASS',cell=cell,trials=20,successes=sum(t['success'] for t in trials),
          seconds=time.monotonic()-begin,checkpoint_sha256=sha(cell['checkpoint']),model_unchanged=True,
          frozen_frontend_hash=frozen,native_evaluator_sha256=sha(ROOT/'libero/lifelong/metric.py'),
          init_file_sha256=sha(init),initial_indices=list(range(20)),formal_training_resumed=False,
          diagnostic_attempt=attempt,
          files={name:sha(dest/name) for name in ('TRIALS.json','ROUNDS.json','CONFIG.json','PAIRING_CHECK.json')}))


def queue(pilot):
    import fcntl
    lock=(OUT/'behavior_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cc=cells(pilot)
    if pilot:
        for unit in ('s45_6_7','s45_7_6'):
            assert (OUT/'signals'/unit/'SEALED.json').exists()
            assert (OUT/'transfers'/unit/'COMPLETE.json').exists()
            assert (OUT/'offline'/unit/'COMPLETE.json').exists()
    freeze(OUT/('PILOT_BEHAVIOR_MANIFEST.json' if pilot else 'BEHAVIOR_MANIFEST.json'),
           dict(cells=cc,total_trials=len(cc)*20,native_action_rule=True,formal_training_resumed=False))
    for index,cell in enumerate(cc):
        if (directory(cell)/'COMPLETE.json').exists():continue
        write(OUT/'STATUS.json',dict(stage='PILOT_BEHAVIOR' if pilot else 'BEHAVIOR',cell=index,
              total_cells=len(cc),current=cell,formal_training_resumed=False))
        log=OUT/('BEHAVIOR_%s_%03d.log'%('pilot' if pilot else 'full',index))
        command=[sys.executable,'-u','-m','task_relation_forecast_v1.behavior','--cell',str(index)]
        if pilot:command.append('--pilot')
        previous=directory(cell)/'STARTED.json'
        failure=directory(cell)/'FAILURE_CONTEXT.json'
        if previous.exists():
            assert failure.exists() and read(failure)['reason']=='Rollout initial pairing mismatch', 'Unclassified interrupted cell'
            returncode=1
        else:
            with log.open('a') as output:
                returncode=subprocess.run(command,cwd=str(ROOT),stdout=output,stderr=subprocess.STDOUT).returncode
        if returncode and failure.exists() and read(failure)['reason']=='Rollout initial pairing mismatch':
            retry=OUT/'behavior_attempts'/('s%d'%cell['seed'])/cell['model_id']/('task%d'%cell['task'])/'attempt001'
            assert not retry.exists(), 'The one permitted isolated retry was already used; stop for diagnosis'
            retry_log=OUT/('BEHAVIOR_retry_%s_%03d.log'%('pilot' if pilot else 'full',index))
            with retry_log.open('a') as output:
                returncode=subprocess.run(command+['--attempt','001'],cwd=str(ROOT),stdout=output,stderr=subprocess.STDOUT).returncode
            if returncode==0:accept_paired_retry(cell,'001')
            else:log=retry_log
        if returncode:
            write(OUT/'STATUS.json',dict(stage='BEHAVIOR_FAILED',cell=cell,log=str(log),
                  returncode=returncode,formal_training_resumed=False))
            write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',stage='behavior',cell=cell,
                  log=str(log),formal_training_resumed=False))
            raise RuntimeError('Native rollout failed; see '+str(log))
    write(OUT/('PILOT_BEHAVIOR_COMPLETE.json' if pilot else 'BEHAVIOR_COMPLETE.json'),
          dict(status='PASS',cells=len(cc),trials=len(cc)*20,formal_training_resumed=False))
    write(OUT/'STATUS.json',dict(stage='PILOT_COMPLETE' if pilot else 'BEHAVIOR_COMPLETE',
          rollouts_completed=len(cc)*20,formal_training_resumed=False))


if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--cell',type=int)
    parser.add_argument('--attempt',help='Isolated diagnostic repetition; never replaces a main result')
    args=parser.parse_args()
    if args.cell is None:
        assert args.attempt is None
        queue(args.pilot)
    else:run_cell(cells(args.pilot)[args.cell],args.attempt)
