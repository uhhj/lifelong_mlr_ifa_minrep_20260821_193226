"""Actual source comparison with historical S, prior q1, and accepted Random."""
from pathlib import Path
import json,hashlib,ast,difflib,sys,time
W=Path('/data/mlr_ifa/experiment_workspaces');Q=W/'MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2'
sys.path.insert(0,str(Q/'engineering_assets_resume_20261110_v2'));import resume_runtime as rt
rt.verify()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text(p):return p.read_text(encoding='utf-8-sig')
def funcs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text(p)).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
out={}
for suite in ['OBJECT','GOAL']:
 for seed in [41,42,43]:
  new=W/f'S_Q1_{suite}41_42_43_20261109_rerun_v1'/f'Q1_{seed}'
  old=W/f'S_Q1_{suite}41_42_43_20261108_v1'/f'Q1_{seed}'
  binding=json.loads(text(new/'S_REFERENCE.json'));base=Path(binding['OUTCOMES.json']['path']).parent
  contract=json.loads(text(new/'CONTRACT.json'));assert all(sha(Path(p))==h for p,h in contract['files'].items())
  refs={k:sha(Path(v['path']))==v['sha256'] for k,v in binding.items()};assert all(refs.values())
  files={}
  for n in ['evaluate.py','collect.py','support_memory.py','losses.py','updates.py','train.py','runtime.py','layout.py','initialization.py','candidate.py','fork_tools.py','gate_admission.py','memory.py','legacy_losses.py','legacy_updates.py']:
   p=new/n;s=base/n;o=old/n
   if not s.exists():
    files[n]=dict(actual_sha256=sha(p),prior_q1_sha256=sha(o),prior_q1_same_text=text(p)==text(o),historical_s_path=str(s),historical_s_exists=False)
    continue
   a,b=funcs(p),funcs(s)
   files[n]=dict(actual_sha256=sha(p),prior_q1_sha256=sha(o),prior_q1_same_text=text(p)==text(o),historical_s_path=str(s),historical_s_sha256=sha(s),historical_s_same_text=text(p)==text(s),same_top_level_functions=[k for k in a if k in b and a[k]==b[k]],different_top_level_functions=[k for k in a if k not in b or a[k]!=b[k]])
   if n in ['evaluate.py','collect.py','support_memory.py','losses.py','updates.py','fork_tools.py','gate_admission.py','memory.py','legacy_losses.py','legacy_updates.py'] and text(p)!=text(s):files[n]['diff']=''.join(difflib.unified_diff(text(s).splitlines(True),text(p).splitlines(True),fromfile='accepted_S/'+n,tofile='q1/'+n))
  out[suite+'_'+str(seed)]=dict(contract_sha256=sha(new/'CONTRACT.json'),reference_hashes=refs,files=files)
print(json.dumps(dict(status='READONLY_ACTUAL_PROTOCOL_SOURCE_AUDIT',observed=time.time(),q1=out,align_contract=sha(Q/'CONTRACT.json'),align_engineering_contract=sha(Q/'engineering_assets_resume_20261110_v2/CONTRACT.json'),new_optimizer=0,new_environment=0)))
