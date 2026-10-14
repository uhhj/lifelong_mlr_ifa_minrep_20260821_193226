"""Read-only seed plumbing and exact seed42 protocol compatibility audit."""
import ast,json,hashlib
from pathlib import Path
from coreset_exp.common import S,O,SEED,SELECTOR_SEED,verify,write
from coreset_exp.configure import make_config
from coreset_exp.model import cfg

base=Path('/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013')
c=verify();assert c['seed']==SEED and SELECTOR_SEED==42
checks=[]
def normalize(x):
    if isinstance(x,dict):return {k:(42 if k=='seed' else normalize(v)) for k,v in x.items()}
    if isinstance(x,list):return [normalize(v) for v in x]
    if isinstance(x,str):return x.replace(f'Coreset_paper_OBJECT_seed{SEED}_20261013','Coreset_paper_OBJECT_seed42_20261013')
    return x
for kind,arm in [('export',None),('train','random'),('train','coreset'),('eval',None)]:
    name=arm or kind
    actual,overrides=make_config(kind,arm)
    assert actual==dict(cfg(name))
    assert normalize(actual)==json.loads((base/f'resolved_{name}.json').read_text()),name
    assert actual['seed']==SEED and f'seed={SEED}' in overrides
    checks.append('resolved_'+name)
for name in ('export','train','preflight','evaluate'):
    tree=ast.parse((S/'coreset_exp'/f'{name}.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='control_seed']
    assert calls and all(isinstance(n.args[0],ast.Name) and n.args[0].id=='SEED' for n in calls)
    checks.append(name+'_control_seed')
source=(S/'coreset_exp/storage.py').read_text()
assert 'random.Random(SEED)' in source and 'greedy_kcenter_select(paths,k,SELECTOR_SEED,population)' in source
assert 'c.buffer_dir,SEED,start=6' in (S/'coreset_exp/train.py').read_text()
assert 'see=SEED' in (S/'coreset_exp/evaluate.py').read_text()
base_source=Path('/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed42_20261013/source')
for name in ('libero/lifelong/models/complete_policy.py','libero/lifelong/models/brief.py','libero/lifelong/algos/base.py','libero/lifelong/algos/mlr.py','libero/lifelong/selectors/greedy_kcenter.py','coreset_exp/eval_metric.py'):
    assert (S/name).read_bytes()==(base_source/name).read_bytes(),name
    checks.append('unchanged_'+name)
write(O/'SEED_EXTENSION_AUDIT.json',dict(status='PASS',seed=SEED,selector_seed=SELECTOR_SEED,checks=checks,config_equivalent_except_seed_and_paths=True))
print('PASS',SEED,len(checks),'seed/config/source checks')
