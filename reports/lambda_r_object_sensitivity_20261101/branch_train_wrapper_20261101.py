"""UNDEPLOYED: change only four within-block snapshots, never checkpoint format."""
import ast,sys,json,hashlib,time,copy
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent
case=sys.argv[1];assert case in ['L005_42','L005_43','L020_41','L020_42','L020_43']
sys.argv=[str(Q/case/'train.py'),'S']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads((E/'CONTRACT.json').read_text())
assert all(sha(p)==h for p,h in contract['files'].items())
v=json.loads((E/'VALIDATION.json').read_text())
assert v['status']=='PASS_PAIRED_FULL_EPOCH_PATH_BOTH_LAMBDAS' and v['optimizer_completed']==v['optimizer_started']==7828 and v['environments']==0
assert (E/'HANDOFF_COMPLETE.json').exists()
sys.path.insert(0,str(Q/case))
import runtime as rt
import train as tr
torch=rt.torch
def device_copy(x):
 if torch.is_tensor(x):return x.detach().clone()
 if isinstance(x,dict):return {k:device_copy(v) for k,v in x.items()}
 if isinstance(x,list):return [device_copy(v) for v in x]
 if isinstance(x,tuple):return tuple(device_copy(v) for v in x)
 return copy.deepcopy(x)
def same_values(x,y):
 if torch.is_tensor(x):return x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride() and torch.equal(x.cpu(),y.cpu())
 if isinstance(x,dict):return x.keys()==y.keys() and all(same_values(x[k],y[k]) for k in x)
 if isinstance(x,(tuple,list)):return type(x)==type(y) and len(x)==len(y) and all(same_values(u,v) for u,v in zip(x,y))
 return rt.equal(x,y)
checked={}
def branch_capture(a):
 original=rt.cpu
 try:
  rt.cpu=device_copy;result=rt._reference_capture(a)
 finally:rt.cpu=original
 if a.step in [15700,38400,69300] and a.scheduler.last_epoch==0 and str(a.step) not in checked:
  oracle=rt.capture(a);assert same_values(oracle,result)
  checked[str(a.step)]=dict(full_state_values_exact=True,optimizer_calls=0,environments=0,time=time.time())
  rt.write(E/(case+'_BRANCH_COPY_EQ.json'),dict(status='PASS_OBSERVED_STAGE_STARTS',checked=checked,contract=sha(E/'CONTRACT.json')))
 return result
source=(Q/case/'train.py').read_text();tree=ast.parse(source,filename=str(Q/case/'train.py'))
main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
class Change(ast.NodeTransformer):
 def __init__(self):self.lines=[]
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='capture' and node.lineno in [93,95,98,100]:
   self.lines.append(node.lineno);node.func.id='branch_capture'
  return node
change=Change();change.visit(main);assert sorted(change.lines)==[93,95,98,100]
tr.branch_capture=branch_capture
exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(Q/case/'train.py'),'exec'),tr.__dict__)
assert tr.capture is rt.capture,'Epoch/checkpoint/admission snapshots remain CPU'
tr.main()
