"""Pure capacity/authorization boundary checks; no SSH, GPU, model or environment."""
import json,importlib.util,sys
from pathlib import Path
# The server runner imports fcntl; use only its pure decision function via AST on Windows.
import ast
p=Path(__file__).parent/'capacity_recovery_v1.py';tree=ast.parse(p.read_text(encoding='utf8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='allowed']
ns={'RESERVE':2*1024**3};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
f=ns['allowed'];e=dict(key='RA_IFA_LIBERO50_43',storage_mount='/data',minimum_free_bytes=4500000000);threshold=e['minimum_free_bytes']+ns['RESERVE'];c=dict(euid=0,free_all_blocks=threshold)
assert f(e,c,False,1,'RESOURCE_CAPACITY')
assert not f(e,dict(c,free_all_blocks=threshold-1),False,1,'RESOURCE_CAPACITY')
assert not f(e,c,True,1,'RESOURCE_CAPACITY')
assert not f(e,c,False,1,'training assertion failure')
assert not f(e,dict(c,euid=1000),False,1,'RESOURCE_CAPACITY')
assert not f(dict(e,storage_mount='/'),c,False,1,'RESOURCE_CAPACITY')
assert not f(dict(e,key='COVR_OBJECT_42'),c,False,1,'RESOURCE_CAPACITY')
assert not f(e,c,False,0,'RESOURCE_CAPACITY')
out=dict(status='CAPACITY_POLICY_BOUNDARIES_PASS',cases=8,scientific_tests=0,optimizer_updates=0,environment_calls=0,server_changes=0)
(Path(__file__).parent/'CAPACITY_POLICY_TEST.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
