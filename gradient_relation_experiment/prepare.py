"""Freeze 120 event-independent origins and full-horizon trajectory exclusion."""
import collections
import gzip
import time
import traceback
import h5py
from cb_ifa_experiment.gpu_runtime import *
from optimizer_signal_diagnostic.prepare import lineage

OUT=ROOT/'reports/gradient_relation_v1'


def epoch_indices(plan,epoch):
    s=torch.utils.data.DistributedSampler(range(len(plan['records'])),**{k:plan['sampler'][k] for k in ('seed','num_replicas','rank','shuffle','drop_last')})
    s.set_epoch(epoch);return list(s)


def batch_indices(plan,step):
    epoch,batch=divmod(step,plan['batches_per_epoch'])
    return epoch_indices(plan,epoch)[batch*10:(batch+1)*10]


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PREPARED.json').exists()
    from libero.libero.benchmark import get_benchmark
    members=read(ROOT/'reports/paired_continuation/isolated_prepared_v1/MEMBERSHIP.json')
    prepared=read(ROOT/'reports/paired_continuation/isolated_prepared_v1/PREPARED.json')
    assert prepared['status']=='PASS'
    original_feature_path=ROOT/'runs/object_seed42_feature/run_001/config.json'
    assert sha(original_feature_path)==prepared['feature_config_sha256']
    original_feature_cfg=read(original_feature_path)
    hashes={};all_points=[];lineages={};plans={}
    for seed in (45,46,47):
        feature_cfg=read(ROOT/f'runs/fixed_kappa_confirmation_v2/feature_seed{seed}/run_001/config.json')
        assert feature_cfg['policy']['use_language'] and not feature_cfg['policy']['use_state']
        assert feature_cfg['policy']==original_feature_cfg['policy'],'Frozen monitor frontend configuration differs'
        assert feature_cfg['data']['demo_limit']==10 and feature_cfg['data']['seq_len']==8
        bench=get_benchmark(feature_cfg['benchmark_name'])(feature_cfg['data']['task_order_index'])
        for task in (6,7,8):
            cfg=dict(feature_cfg,feature_dir=str(ROOT/f'features/object_seed{seed}'))
            raw=ROOT/'data'/bench.get_task_demonstration(task)
            assert sha(raw)==members['raw_sha256'][str(raw.relative_to(ROOT))]
            records=lineage(ROOT,cfg,task,raw,hashes)
            lineages[seed,task]={r['path']:r['demos'] for r in records}
        for task in (7,8):
            path=ROOT/f'runs/fixed_kappa_confirmation_v2/formal/seed{seed}/mlr/plans/task{task}.json'
            plan=read(path);plans[f'{seed}/{task}']={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
            assert plan['num_workers']==0 and plan['epochs']==100 and plan['batch_size']==10
            for origin,h in zip(plan['immutable_feature_origins'],plan['record_sha256']):assert hashes[origin]==h
            bpe=plan['batches_per_epoch'];origins=plan['immutable_feature_origins']
            for lo,hi in ((30,70),(70,100)):
                rng=np.random.RandomState(20260913+seed*1000+task*100+lo)
                left=lo*bpe;right=min(hi*bpe,100*bpe-19)
                edges=np.linspace(left,right,11,dtype=int)
                for slot in range(10):
                    step=int(rng.randint(edges[slot],edges[slot+1]));epoch,batch=divmod(step,bpe)
                    window=[batch_indices(plan,step+h) for h in range(20)]
                    ids=[int(Path(origins[i]).parent.name) for i in window[0]]
                    refs={};used={t:set() for t in range(6,task+1)}
                    for row in window:
                        for i in row:
                            t=int(Path(origins[i]).parent.name);used[t].update(lineages[seed,t][origins[i]])
                    train_used={str(t):sorted(v) for t,v in used.items()}
                    ref_used={t:set() for t in used}
                    for old in sorted(t for t in set(ids) if t<task):
                        pool=[i for i,o in enumerate(origins) if int(Path(o).parent.name)==old]
                        count=ids.count(old)
                        refs[str(old)]=[rng.choice(pool,count,replace=False).tolist() for _ in range(3)]
                        for rr in refs[str(old)]:
                            for i in rr:ref_used[old].update(lineages[seed,old][origins[i]])
                    for t in used:
                        used[t]|=ref_used[t]
                        for bank in ('A','B'):
                            chosen={x['demo'] for x in members['membership'][str(t)][bank]}
                            assert not used[t]&chosen,('Trajectory isolation unavailable',seed,task,step,t,bank)
                    all_points.append({'point':f's{seed}_t{task}_e{epoch}_b{batch}','seed':seed,'task':task,
                        'phase':f'{lo:02d}-{hi-1:02d}','slot':slot,'epoch':epoch,'batch':batch,'stage_step':step,
                        'future20_record_indices':window,'first_task_ids':ids,'stability_references':refs,
                        'training_trajectory_union':train_used,'reference_trajectory_union':{str(t):sorted(v) for t,v in ref_used.items()},
                        'combined_trajectory_union':{str(t):sorted(v) for t,v in used.items()},
                        'new_present':task in ids,'old_present':sorted(t for t in set(ids) if t<task)})
    assert len(all_points)==120 and len({p['point'] for p in all_points})==120
    monitor={str(t):members['membership'][str(t)] for t in (6,7,8)}
    for banks in monitor.values():
        for rows in banks.values():
            for row in rows:
                p=ROOT/row['path'];assert sha(p)==prepared['feature_sha256'][row['path']];hashes[row['path']]=sha(p)
    assert sha(ROOT/'checkpoints/object_multitask_model_ep99.pth')==prepared['checkpoint_sha256']
    frozen={'version':'gradient_relation_v1','frozen_at':time.time(),'seeds':[45,46,47],'target_points':120,
        'horizons':[1,5,20],'primary_horizon':20,'source_plan_sha256':plans,'monitor_membership':monitor,
        'monitor_prepared_sha256':sha(ROOT/'reports/paired_continuation/isolated_prepared_v1/PREPARED.json'),
        'points':all_points,'sample_scope':'stratified event-independent positions; not selected on cosine/trigger/outcome',
        'isolation_scope':'complete trajectory unused in these20 updates and all auxiliary references; no historical unseen claim',
        'model_selection':'none; LOSO Ridge alpha1 and LogisticRegression C1; no random window split',
        'input_sha256':hashes}
    with gzip.open(OUT/'DESIGN.json.gz','wt',encoding='utf-8') as f:json.dump(frozen,f)
    write(OUT/'PREPARED.json',{'status':'PASS','time':time.time(),'points':120,'design_sha256':sha(OUT/'DESIGN.json.gz'),
        'monitor_windows':120,'new_absent_points':sum(not p['new_present'] for p in all_points),
        'task_pair_observations':sum(len(p['old_present']) for p in all_points),'future20_trajectory_isolation':True})
    print('PREPARED',read(OUT/'PREPARED.json'),flush=True)


if __name__=='__main__':
    try:run()
    except Exception:
        write(OUT/'PREPARATION_FAILED.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
