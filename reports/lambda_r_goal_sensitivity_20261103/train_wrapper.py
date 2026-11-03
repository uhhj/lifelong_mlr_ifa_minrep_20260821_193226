"""Fresh GOAL run: reviewed device snapshots and exact predicate checks, no resume."""
import ast,copy,inspect,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
case=sys.argv[1]
assert case in [g+'_'+str(s) for g in ['L005','L020'] for s in (41,42,43)]
sys.path.insert(0,str(ROOT/case));sys.argv=[str(ROOT/case/'train.py'),'S']
import runtime as rt
import train as tr
import torch
contract=rt.js(ROOT/'CONTRACT.json')
assert all(rt.sha(p)==h for p,h in contract['files'].items())
validation=rt.js(ROOT/'ENGINEERING_VALIDATION.json')
assert validation['status']=='PASS_GOAL_FULL_PATH_BOTH_LAMBDAS'
assert validation['optimizer_started']==validation['optimizer_completed']==5704
assert len(validation['stages'])==8 and validation['environments']==0
assert os.environ['WORLD_SIZE']=='1' and os.environ['RANK']=='0'
rt.write(rt.R/'RENDEZVOUS.json',dict(pid=os.getpid(),time=time.time(),master_addr=os.environ['MASTER_ADDR'],master_port=os.environ['MASTER_PORT'],world_size=1))

def device(x):
 if torch.is_tensor(x):return x.detach().clone()
 if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
 if isinstance(x,list):return [device(v) for v in x]
 if isinstance(x,tuple):return tuple(device(v) for v in x)
 return copy.deepcopy(x)
checked={};finite_starts={};device_starts={}
def branch_capture(a):
 original=rt.cpu
 try:rt.cpu=device;state=rt._reference_capture(a)
 finally:rt.cpu=original
 if a.scheduler.last_epoch==0 and str(a.step) not in checked:
  assert rt.equal(rt.cpu(state),rt.capture(a))
  checked[str(a.step)]=dict(all_state_exact=True,optimizer_calls_added=0,environments_added=0)
  rt.write(rt.R/'BRANCH_COPY_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=checked))
 return state
def checks_write():
 rt.write(rt.R/'CHECK_EQ.json',dict(status='PASS_OBSERVED_CHECKS',finite_stage_starts=finite_starts,device_stage_starts=device_starts,optimizer_calls_added=0,environments_added=0))
def combined_finite(a,loss):
 pars=list(a.policy.parameters())
 result=bool(torch.stack([torch.isfinite(loss).all()]+[torch.isfinite(p.grad).all() for p in pars if p.grad is not None]).all())
 if a.scheduler.last_epoch==0 and str(a.step) not in finite_starts:
  branch_capture(a)
  oracle=bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pars)
  assert result==oracle
  finite_starts[str(a.step)]=dict(exact=True);checks_write()
 return result
def device_check(a,selected):
 result=rt.equal(selected,branch_capture(a))
 if a.scheduler.last_epoch<=10 and str(a.step) not in device_starts:
  assert result==rt.equal(rt.cpu(selected),rt.cpu(rt.capture(a)))
  device_starts[str(a.step)]=dict(exact=True);checks_write()
 return result
source=inspect.getsource(tr.update)
needle='assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())'
assert source.count(needle)==1
namespace=dict(tr.update.__globals__,combined_finite=combined_finite)
exec(compile(source.replace(needle,'assert combined_finite(a,loss)'),__file__,'exec'),namespace)
tr.update=namespace['update']
main=ast.parse(inspect.getsource(tr.main)).body[0]
class Change(ast.NodeTransformer):
 def __init__(self):self.copies=0;self.checks=0
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['base','cstate','qstate'] and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='capture':
   node.value.func.id='branch_capture';self.copies+=1
  return self.generic_visit(node)
 def visit_Assert(self,node):
  if ast.dump(node.test)==ast.dump(ast.parse('equal(selected,capture(a))',mode='eval').body):
   node.test=ast.copy_location(ast.parse('device_check(a,selected)',mode='eval').body,node.test);self.checks+=1
  return self.generic_visit(node)
change=Change();change.visit(main);assert change.copies==3 and change.checks==1
tr.branch_capture=branch_capture;tr.device_check=device_check
exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),__file__,'exec'),tr.__dict__)
assert tr.capture is rt.capture
tr.main()
