"""UNDEPLOYED: exact gate-state checks on device and batched finite predicates."""
import sys,json,hashlib
from pathlib import Path
CHECK_E=Path(__file__).resolve().parent;Q=CHECK_E.parent
PREV=Q/'engineering_resume_3090_20261102_v1'
case=sys.argv[1]
assert case in ['L005_42','L005_43','L020_41','L020_42','L020_43']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((CHECK_E/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
v=json.loads((CHECK_E/'VALIDATION.json').read_text())
assert v['status']=='PASS_3090_CHECKS_FULL_EPOCH_PATH_BOTH_LAMBDAS' and v['optimizer_started']==v['optimizer_completed']==7828
assert len(v['stages'])==8 and v['environments']==0 and (CHECK_E/'HANDOFF_COMPLETE.json').exists()
source=(PREV/'train_wrapper.py').read_text()
assert source.count('tr.main()')==1
hook='''
import inspect
finite_starts={};device_starts={}
def write_checks():
 rt.write(CHECK_E/(case+'_CHECK_EQ.json'),dict(status='PASS_OBSERVED_CHECKS',finite_stage_starts=finite_starts,device_stage_starts=device_starts,optimizer_calls_added=0,environments_added=0))
def combined_finite(a,loss):
 pars=list(a.policy.parameters())
 result=bool(torch.stack([torch.isfinite(loss).all()]+[torch.isfinite(p.grad).all() for p in pars if p.grad is not None]).all())
 if a.scheduler.last_epoch==0 and str(a.step) not in finite_starts:
  oracle=bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pars)
  assert result==oracle
  finite_starts[str(a.step)]=dict(exact=True,time=time.time());write_checks()
 return result
update_source=inspect.getsource(tr.update)
needle='assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())'
assert update_source.count(needle)==1
candidate_globals=dict(tr.update.__globals__,combined_finite=combined_finite)
exec(compile(update_source.replace(needle,'assert combined_finite(a,loss)'),str(CHECK_E/'train_wrapper.py'),'exec'),candidate_globals)
tr.update=candidate_globals['update']
def device_check(a,selected):
 observed=branch_capture(a)
 result=rt.equal(selected,observed)
 if a.scheduler.last_epoch<=10 and str(a.step) not in device_starts:
  oracle=rt.equal(rt.cpu(selected),rt.cpu(rt.capture(a)))
  assert result==oracle
  device_starts[str(a.step)]=dict(exact=True,time=time.time());write_checks()
 return result
class CheckChange(ast.NodeTransformer):
 def __init__(self):self.count=0
 def visit_Assert(self,node):
  if ast.dump(node.test,include_attributes=False)==ast.dump(ast.parse('equal(selected,capture(a))',mode='eval').body,include_attributes=False):
   node.test=ast.copy_location(ast.parse('device_check(a,selected)',mode='eval').body,node.test);self.count+=1
  return self.generic_visit(node)
checks_change=CheckChange();checks_change.visit(main);assert checks_change.count==1
tr.device_check=device_check
exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(CHECK_E/'train_wrapper.py'),'exec'),tr.__dict__)
assert tr.capture is rt.capture,'CPU checkpoint/admission/collection parity unchanged'
tr.main()
'''
# Preserve the previously verified recovery/hardware wrapper and its own provenance files.
scope=dict(__file__=str(PREV/'train_wrapper.py'),__name__='__main__',CHECK_E=CHECK_E)
exec(compile(source.replace('tr.main()',hook),str(CHECK_E/'train_wrapper.py'),'exec'),scope)
