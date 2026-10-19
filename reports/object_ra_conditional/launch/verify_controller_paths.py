"""Synthetic control-flow verification only; no model, optimizer or environment."""
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])))
import controller as c
cases=[]
for first,second,expected in [(True,True,['OBJECT_RA','GOAL_RA']),(True,False,['OBJECT_RA']),(False,False,['GOAL_Coreset'])]:
 with tempfile.TemporaryDirectory() as td:
  c.N=Path(td);calls=[];c.verify_controller=lambda:None;c.no_workers=lambda:None
  c.run_stage1=lambda:dict(gate=dict(passed=first))
  def group(branch,seeds):
   calls.append(branch)
   assert seeds==([41,42,43,44] if branch=='GOAL_Coreset' else [41,42,43])
   return dict(gate=dict(passed=second))
  c.group_zr=group;c.main()
  assert calls==expected and (c.N/'controller.exit').read_text().strip()=='0'
  cases.append(dict(first=first,second=second,calls=calls,status='PASS'))
with tempfile.TemporaryDirectory() as td:
 c.N=Path(td);calls=[]
 def broken():raise RuntimeError('synthetic engineering failure')
 c.run_stage1=broken
 try:c.main()
 except RuntimeError:pass
 else:raise AssertionError('Engineering failure was swallowed')
 assert calls==[] and (c.N/'controller.exit').read_text().strip()=='1'
 assert json.loads((c.N/'FAILURE.json').read_text())['performance_fallback_on_engineering_failure'] is False
 cases.append(dict(engineering_failure='STOP_WITHOUT_FALLBACK',status='PASS'))
print(json.dumps(dict(status='PASS',synthetic=True,gpu_updates=0,environment_steps=0,cases=cases),indent=2))
