"""Run frozen seed43 train.main with only the CPU snapshot primitive replaced."""
import sys,json,hashlib,time
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((E/'CONTRACT.json').read_text())
assert all(sha(p)==h for p,h in c['files'].items())
assert json.loads((E/'VALIDATION.json').read_text())['status']=='PASS_FULL_TRAINING_PATH_EXACT'
assert (E/'HANDOFF_COMMITTED.json').exists()
sys.path.insert(0,str(Q/'seed43'))
import runtime as rt
from candidate import cpu_snapshot
old_cpu=rt.cpu;old_capture=rt.capture;checked={}
def capture_checked(a):
 result=old_capture(a)
 # Check actual seed43 full state at each new-stage optimizer initialization.
 key=str(a.step)
 if a.step in [0,15700,38400,69300] and a.scheduler.last_epoch==0 and key not in checked:
  try:
   rt.cpu=old_cpu;oracle=old_capture(a)
  finally:rt.cpu=cpu_snapshot
  assert rt.equal(result,oracle),'Optimized capture differs on seed43 actual state'
  checked[key]=dict(time=time.time(),all_state_exact=True,optimizer_calls_added=0,environments_added=0)
  rt.write(E/'SEED43_CAPTURE_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=checked,candidate_sha256=sha(E/'candidate.py'),engineering_contract_sha256=sha(E/'CONTRACT.json')))
 return result
rt.cpu=cpu_snapshot;rt.capture=capture_checked
import train
assert train.cpu is cpu_snapshot and train.capture is capture_checked
assert rt.verify()==c['seed43_contract_sha256']
train.main()
