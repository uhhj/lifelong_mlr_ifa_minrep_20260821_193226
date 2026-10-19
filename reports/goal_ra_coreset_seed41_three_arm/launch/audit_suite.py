"""Read-only GOAL resource, resolved-config and source equivalence gate. No CUDA allocation."""
import sys,json,hashlib,math,ast,os
from pathlib import Path
import h5py,numpy as np,torch
from coreset_exp.common import S,O,SEED,write,sha,source_hashes
from coreset_exp.configure import make_config
from libero.libero.benchmark import get_benchmark
Q=Path(__file__).resolve().parent
for n,h in json.loads((S.parent/'SOURCE_SHA256.json').read_text()).items():assert sha(S/n)==h,n
P=Path('/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed43_20261013/source')
parent=json.loads(Path('/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed43_20261013/CONTRACT.json').read_text())
for n,h in parent['source_sha256'].items():assert sha(P/n)==h,n
checks=[]
for n in ('libero/lifelong/models/complete_policy.py','libero/lifelong/models/brief.py','libero/lifelong/algos/base.py','libero/lifelong/algos/mlr.py','libero/lifelong/selectors/greedy_kcenter.py','coreset_exp/eval_metric.py','coreset_exp/storage.py'):
    assert (S/n).read_bytes()==(P/n).read_bytes(),n;checks.append('unchanged:'+n)
configs={};commands={}
for kind,arm in [('export',None),('train','random'),('train','coreset'),('eval',None)]:
    c,overrides=make_config(kind,arm);name=arm or kind
    assert c['benchmark_name']=='LIBERO_GOAL' and c['seed']==SEED and c['gamma']==.7
    assert c['train']['n_epochs']==100 and c['train']['batch_size']==(1 if kind=='export' else 10)
    assert c['eval']['n_eval']==20 and c['eval']['max_steps']==600
    assert c['data']['seq_len']==8 and c['data']['demo_limit']==10 and c['data']['task_order_index']==0
    assert all(type(c['policy'][k]) is bool and c['policy'][k] for k in ('use_vision','use_language','use_state'))
    # Only suite, explicit gamma, official weight, training seed and isolated paths differ from OBJECT.
    old=json.loads(Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed43_20261013/resolved_{name}.json').read_text())
    norm=json.loads(json.dumps(c).replace(str(O),'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed43_20261013'))
    norm.update(seed=43,benchmark_name='LIBERO_OBJECT',gamma=.3,pretrain=old['pretrain'])
    assert norm==old,('Unexpected resolved config difference',name)
    configs[name]=c;commands[name]=overrides
    preview=S.parent/'config_preview';write(preview/f'resolved_{name}.json',c)
    if (O/'CONTRACT.json').exists():assert c==json.loads((O/f'resolved_{name}.json').read_text())
for name in ('export','train','preflight','evaluate'):
    tree=ast.parse((S/'coreset_exp'/f'{name}.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='control_seed']
    assert calls and all(isinstance(n.args[0],ast.Name) and n.args[0].id=='SEED' for n in calls)
b=get_benchmark('LIBERO_GOAL')(0);inputs={};resources={}
for t in range(10):
    task=b.get_task(t);demo=Path(configs['export']['folder'])/b.get_task_demonstration(t)
    bd=Path(configs['export']['bddl_folder'])/task.problem_folder/task.bddl_file
    ini=Path(configs['export']['init_states_folder'])/task.problem_folder/task.init_states_file
    for p in (demo,bd,ini):assert p.is_file(),p;inputs[str(p)]=sha(p)
    with h5py.File(demo,'r') as f:
        demos=sorted(f['data'].keys(),key=lambda x:int(x.split('_')[-1]))[:10]
        assert len(demos)==10
        lens=[int(f['data'][d].attrs['num_samples']) for d in demos]
        for d,n in zip(demos,lens):
            assert f['data'][d]['actions'].shape==(n,7)
            assert f['data'][d]['obs']['joint_states'].shape==(n,7)
            assert f['data'][d]['obs']['gripper_states'].shape==(n,2)
            for k in ('agentview_rgb','eye_in_hand_rgb'):assert f['data'][d]['obs'][k].shape[0]==n
    states=torch.load(ini,map_location='cpu');assert len(states)>=20
    digests=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in states[:20]];assert len(set(digests))==20
    resources[str(t)]=dict(name=task.name,demos=demos,windows=sum(lens),initial_count=len(states),first20_sha256=digests,paths=list(map(str,(demo,bd,ini))))
counts=[resources[str(t)]['windows'] for t in range(6,10)];assert counts==[997,889,904,1969],counts
quotas=[math.ceil(n*.5) for n in counts];batches=[math.ceil((n+sum(quotas[:i]))/10) for i,n in enumerate(counts)]
assert batches==[100,139,185,337] and 100*sum(batches)==76100
pt=Path(configs['export']['pretrain']);inputs[str(pt)]=sha(pt)
assert inputs[str(pt)]=='241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7'
cp=torch.load(pt,map_location='cpu');policy=cp.get('cfg',{}).get('policy',{})
sd=cp.get('state_dict',cp.get('model',{}));assert sd
state_params={k:list(v.shape) for k,v in sd.items() if any(x in k for x in ('state_encoder.','state_scale.','state_shift.'))}
assert len(state_params)==12 and all(torch.isfinite(v).all() for k,v in sd.items() if k in state_params)
old=Path('/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/source')
assert 'control_seed(42)' in (old/'libero/lifelong/runbook_v2.py').read_text()
old_hash={str(old/n):sha(old/n) for n in ('libero/lifelong/runbook_v2.py','libero/lifelong/runbook_core.py','scripts/goal_v2_driver.py')}
result=dict(status='PASS_CPU_READONLY_NOT_GPU',seed=SEED,resources=resources,input_sha256=inputs,
    windows=counts,quotas=quotas,batches_per_epoch=batches,steps_per_arm=76100,checks=checks,
    pretrain_use_state=policy.get('use_state','unknown'),effective_use_state=True,state_params=state_params,
    configs=configs,actual_hydra_overrides=commands,source_sha256=source_hashes(),historical_source_sha256=old_hash,
    baseline_reused=False,reason='Different historical evaluation RNG reset/export workers/IFA reference and angle implementation; old result remains valid historical evidence, not exact paired control.')
write(S.parent/'SUITE_RESOURCE_AUDIT.json',result)
if (O/'CONTRACT.json').exists():
    from coreset_exp.common import verify
    verify();write(O/'SUITE_RESOURCE_AUDIT.json',result)
print('PASS CPU resource/config audit',SEED,'4759 windows, 76100 steps/arm; GPU preflight pending or separately recorded')
