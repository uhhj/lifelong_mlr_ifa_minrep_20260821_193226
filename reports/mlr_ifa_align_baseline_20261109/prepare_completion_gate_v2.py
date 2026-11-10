"""Local-only adaptation of the completed q1 engineering exit paths; no runs."""
from pathlib import Path
import ast, json, hashlib, difflib
R=Path(__file__).resolve().parent
D=R/'engineering_completion_gate_20261110_v2'
assert not D.exists(), 'Preserve the previous preparation; inspect instead of overwriting'
D.mkdir()
sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
old=json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'))
c=json.loads(old['CONTRACT.json'])
assert sha(old['CONTRACT.json'])=='6a6298283056986941016d1f487f8169fcbfe760679088e675f51ceaa015f7d7'
assert all(sha(old[n])==h for n,h in c['code_files'].items())
assert all(sha(old['private/'+n])==h for n,h in c['private_files'].items())
s=old['runtime.py'];start=s.index('def previous_finished():');end=s.index('def acceptance_gate():',start)
new='''def previous_finished():
 assert sha(OLD/'CONTRACT.json')==OLD_HASH
 assert all(sha(path)==h for path,h in js(OLD/'CONTRACT.json')['files'].items())
 eng=OLD/'engineering_reboot_resume_20261109_v1'
 assert sha(eng/'CONTRACT.json')=='9c5b7e0b23e1beabb4bf28216f6a75f372d6d2f68b3f6515c27debdd8eda34bb'
 assert all(sha(path)==h for path,h in js(eng/'CONTRACT.json')['files'].items())
 assert not (OLD/'controller.exit').exists()
 assert (eng/'controller.exit').read_text().strip()=='0'
 roots=[OLD]
 for suite in ['OBJECT','GOAL']:
  root=OLD.parent/('S_Q1_'+suite+'41_42_43_20261109_rerun_v1');roots.append(root)
  exit_path=eng/'OBJECT_controller.exit' if suite=='OBJECT' else root/'controller.exit'
  if suite=='OBJECT':assert not (root/'controller.exit').exists()
  assert exit_path.read_text().strip()=='0'
  assert js(root/'Q1_EVALUATION_COMPLETE.json')['episodes']==600
  for seed in [41,42,43]:
   d=root/('Q1_'+str(seed))
   train_exit=eng/('Q1_'+str(seed)+'_train.exit') if suite=='OBJECT' and seed in [41,42] else d/'train_S.exit'
   assert train_exit.read_text().strip()=='0' and (d/'evaluate.exit').read_text().strip()=='0'
   assert len(js(d/'EPOCHS_S.json'))==400 and len(js(d/'OUTCOMES.json'))==200
 assert not processes(roots),'Current q=1 still active; never interrupt it'
'''
runtime=s[:start]+new+s[end:];ast.parse(runtime)
bundle=dict(old);bundle['runtime.py']=runtime
c['code_files']['runtime.py']=sha(runtime)
bundle['CONTRACT.json']=json.dumps(c,indent=2)+'\n'
changed=[n for n in old if old[n]!=bundle[n]]
assert changed==['runtime.py','CONTRACT.json'] or set(changed)=={'runtime.py','CONTRACT.json'}
for n,h in c['code_files'].items():assert sha(bundle[n])==h
for n,h in c['private_files'].items():assert sha(bundle['private/'+n])==h
for n,t in bundle.items():
 if n.endswith('.py'):ast.parse(t)
(D/'runtime.py').write_text(runtime,encoding='utf8')
(D/'PREPARED_CONTRACT.json').write_text(bundle['CONTRACT.json'],encoding='utf8')
(D/'DEPLOY_TEXT_BUNDLE.json').write_text(json.dumps(bundle),encoding='utf8')
(D/'runtime.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),runtime.splitlines(True),fromfile='prepared/runtime.py',tofile='completion_v2/runtime.py')),encoding='utf8')
# The local observation/final entry points bind the actual new text contract.
for n in ['check_readonly.py','final_readonly.py']:
 text=(R/n).read_text(encoding='utf8')
 text=text.replace(sha(old['CONTRACT.json']),sha(bundle['CONTRACT.json']))
 (D/n).write_text(text,encoding='utf8')
transport=(R/'transport.py').read_text(encoding='utf8').replace("str(R.parent/'20261027_support_gate_forks')","str(R.parents[1]/'20261027_support_gate_forks')")
(D/'transport.py').write_text(transport,encoding='utf8')
launch=(R/'launch_after_q1.py').read_text(encoding='utf8')
launch=launch.replace("R=Path(__file__).resolve().parent;W=R.parents[1];OLD=R.parent/'20261109_support_calibration_rerun';G=W/'publish_mlr_branch'", "R=Path(__file__).resolve().parent;BASE=R.parent;W=BASE.parents[1];OLD=BASE.parent/'20261109_q1_user_resume';G=W/'publish_mlr_branch'")
launch=launch.replace("'reports/support_calibration_rerun_20261109/'", "'reports/q1_user_resume_20261109/'")
launch=launch.replace("d.relative_to((OLD/suite).resolve())", "d.relative_to(OLD.resolve())")
launch=launch.replace("str(R.parent/'20261027_support_gate_forks')", "str(BASE.parent/'20261027_support_gate_forks')")
(D/'launch_after_q1.py').write_text(launch,encoding='utf8')
for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
review=dict(status='PASS_LOCAL_STATIC_COMPLETION_SOURCE_ADAPTATION_NOT_DEPLOYED',original_contract=sha(old['CONTRACT.json']),adapted_contract=sha(bundle['CONTRACT.json']),changed_bundle_files=changed,all_case_training_evaluation_inputs_exact=True,all_private_metadata_exact=True,controller_exact=True,new_optimizer=0,new_environments=0,required_real_global_exit='engineering_reboot_resume_20261109_v1/controller.exit0',legacy_exits_must_remain_absent=True)
(D/'REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
print(json.dumps(review))
