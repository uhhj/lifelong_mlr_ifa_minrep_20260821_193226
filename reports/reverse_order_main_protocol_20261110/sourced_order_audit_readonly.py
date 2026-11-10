"""Read exact inherited factory code without importing models or environments."""
from pathlib import Path
import ast,hashlib,json,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering04')
spec=json.loads((Q/'SPEC.json').read_text());result={}
for case,meta in spec['cases'].items():
    r=Q/'cases'/case;inp=json.loads((r/'INPUT.json').read_text())
    root=Path(inp['source_root'])
    files=[root/'caet_v3_exp/train.py',root/'coreset_exp/model.py'] if meta['method']=='DEBR' else [root/'coreset_exp/train.py',root/'coreset_exp/model.py']
    proof={}
    for p in files:
        raw=p.read_bytes();text=raw.decode('utf-8-sig');nodes=ast.parse(text).body
        functions={n.name:ast.get_source_segment(text,n) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in ['build','data_setup']}
        proof[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),functions=functions)
    result[case]=proof
print(json.dumps(dict(status='READONLY_EXACT_NATIVE_FACTORY_ORDER_AUDIT',observed=time.time(),cases=result,new_import=0,new_optimizer=0,new_environment=0,server_writes=0)))
