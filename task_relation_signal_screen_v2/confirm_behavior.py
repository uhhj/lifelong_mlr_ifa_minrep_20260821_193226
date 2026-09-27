"""40 paired native trials per unique frozen confirmation model/task cell."""
import argparse
import os
import time
from pathlib import Path
from .common import ROOT,read,write,freeze,sha,stable_seed
from .confirm_runtime import REPORT as OUT,model_path,entry
from cb_ifa_experiment.gpu_runtime import bind
bind()

def cells():
    result=[]
    for seed in (48,49,50):
        def add(kind,task,source=None,target=None):
            if kind=='base':mid='base';path=model_path(seed,kind)
            elif kind in ('source','reference'):
                mid=kind+str(source);path=model_path(seed,kind,source,step=1000 if kind=='source' else 300)
            else:mid='transfer%d_%d'%(source,target);path=model_path(seed,kind,source,target,300)
            result.append(dict(seed=seed,model_id=mid,kind=kind,source=source,target=target,task=task,checkpoint=str(path),environment_seed=seed,round_policy_seeds=[stable_seed('task_relation_signal_screen_v2','rollout',seed,task,r) for r in range(4)]))
        for task in range(6):
            add('base',task);add('source',task,task);add('reference',task,task)
        for source in range(6):
            for target in range(6):
                if source!=target:
                    for task in (source,target):add('transfer',task,source,target)
    assert len(result)==234
    return result

def directory(cell):return OUT/'behavior'/('s%d'%cell['seed'])/cell['model_id']/('task%d'%cell['task'])

def policy_input_view(obs,cfg):
    names=[name for values in cfg.data.obs.modality.values() for name in values]
    return [{name:row[cfg.data.obs_key_mapping[name]] for name in names} for row in obs]

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
    cfg = eval_config(45, dest)
    cfg.seed=cell['seed'];cfg.benchmark_name=entry(cell['task'])['suite']
    cfg.eval.n_eval=40
    native_task=entry(cell['task'])['index']
    assert cfg.eval.n_eval == 40
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
            from paper_qp_rollout_v1.common import observation_digest
            index=len(self.rounds)
            policy_digest=observation_digest(policy_input_view(obs,cfg))
            self.active['policy_input_obs_sha256']=policy_digest
            policy_shared=OUT/'behavior/policy_input_pairing'/('s%d_t%d_r%d.json'%(cell['seed'],cell['task'],index))
            if not policy_shared.exists():
                freeze(policy_shared,dict(sha256=policy_digest,first_cell=cell,
                       raw_keys=[cfg.data.obs_key_mapping[n] for values in cfg.data.obs.modality.values() for n in values],
                       provenance='Native configured observation fields only; historical full-observation hashes retained separately'))
            policy_match=read(policy_shared)['sha256']==policy_digest
            shared=OUT/'behavior'/'pairing'/('s%d_t%d.json'%(cell['seed'],cell['task']))
            if shared.exists():
                import numpy as np
                expected=read(shared);index=len(self.rounds)
                actual=dict(init=self.active['init_sha256'],initial_obs=self.active['initial_obs_sha256'],
                            policy_rng_before=self.active['policy_rng_before'])
                checks={k:actual[k]==expected[k][index] for k in ('init','policy_rng_before')}
                checks['policy_input_obs']=policy_match
                write(dest/('INITIAL_PAIRING_round%d.json'%index),dict(checks=checks,actual=actual,
                      expected={k:expected[k][index] for k in actual},policy_input_sha256=policy_digest,
                      full_observation_match=actual['initial_obs']==expected['initial_obs'][index],
                      formal_training_resumed=False))
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
        values = metric.evaluate_success(cfg, algo, bench, [native_task], None, see=cell['seed'])
    finally:
        metric.SubprocVectorEnv = factory
    rounds = [r for observer in instances for r in observer.rounds]
    assert len(instances)==1 and len(rounds)==4
    trials = [dict(seed=cell['seed'],model_id=cell['model_id'],task=cell['task'],
                   trial=r['round']*10+i,success=int(v),first_success_step=r['first'][i],policy_steps=r['count'])
              for r in rounds for i,v in enumerate(r['success'])]
    assert len(trials)==40 and abs(float(values[0])-sum(t['success'] for t in trials)/40)<1e-12
    assert before == digest(algo.policy.state_dict())
    task = bench.get_task(native_task)
    init = Path(cfg.init_states_folder)/task.problem_folder/task.init_states_file
    assert len(torch.load(init))>=40
    # The same seed/task must have identical initialization and round RNG across all models.
    pair_evidence = dict(init=[r['init_sha256'] for r in rounds],
                         initial_obs=[r['initial_obs_sha256'] for r in rounds],
                         policy_rng_before=[r['policy_rng_before'] for r in rounds])
    shared = OUT/'behavior'/'pairing'/('s%d_t%d.json'%(cell['seed'],cell['task']))
    # Persist the actual completed rounds before any pairing assertion. Diagnostics
    # must never discard the first mismatching fields together with the result.
    write(dest/'ROUNDS.json',rounds);write(dest/'TRIALS.json',trials)
    expected_pair=read(shared) if shared.exists() else pair_evidence
    pairing_checks={key:expected_pair[key]==pair_evidence[key] for key in ('init','policy_rng_before')}
    pairing_checks['policy_input_obs']=all(r['policy_input_obs_sha256']==read(OUT/'behavior/policy_input_pairing'/('s%d_t%d_r%d.json'%(cell['seed'],cell['task'],r['round'])))['sha256'] for r in rounds)
    write(dest/'PAIRING_CHECK.json',dict(status='PASS' if all(pairing_checks.values()) else 'FAIL',
          checks=pairing_checks,expected=expected_pair,actual=pair_evidence,
          full_observation_match=expected_pair['initial_obs']==pair_evidence['initial_obs'],
          source_pairing_sha256=sha(shared) if shared.exists() else None,
          diagnostic_attempt=attempt,formal_training_resumed=False))
    if not all(pairing_checks.values()):
        write(dest/'FAILURE_CONTEXT.json',dict(reason='Rollout initial pairing mismatch',
              cell=cell,checks=pairing_checks,checkpoint_sha256=sha(cell['checkpoint']),
              native_evaluator_sha256=sha(ROOT/'libero/lifelong/metric.py'),
              completed_rollouts_retained=40,formal_training_resumed=False))
        raise AssertionError('Rollout initial pairing mismatch: '+str(pairing_checks))
    if not shared.exists():freeze(shared,pair_evidence)
    write(dest/'COMPLETE.json',dict(status='PASS',cell=cell,trials=40,successes=sum(t['success'] for t in trials),
          seconds=time.monotonic()-begin,checkpoint_sha256=sha(cell['checkpoint']),model_unchanged=True,
          frozen_frontend_hash=frozen,native_evaluator_sha256=sha(ROOT/'libero/lifelong/metric.py'),
          init_file_sha256=sha(init),initial_indices=list(range(40)),formal_training_resumed=False,
          diagnostic_attempt=attempt,
          files={name:sha(dest/name) for name in ('TRIALS.json','ROUNDS.json','CONFIG.json','PAIRING_CHECK.json')}))


if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    p=argparse.ArgumentParser();p.add_argument('--cell',type=int,required=True);a=p.parse_args()
    manifest=OUT/'BEHAVIOR_MANIFEST.json'
    freeze(manifest,dict(cells=cells(),total_trials=9360,native_action_rule=True,formal_training_resumed=False))
    run_cell(cells()[a.cell])
