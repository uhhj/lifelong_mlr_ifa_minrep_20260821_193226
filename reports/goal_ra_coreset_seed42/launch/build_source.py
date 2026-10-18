"""Build a new RA workspace; explicit latest-user legacy protocol override."""
from pathlib import Path
import shutil,json,hashlib,difflib
R=Path(__file__).resolve().parent;B=R.parent/'20261018_scv2_goal_seed42';S=R/'source'
assert not S.exists()
shutil.copytree(B/'source',S,ignore=shutil.ignore_patterns('__pycache__'))
(S/'sc_exp').rename(S/'ra_exp')
for p in (S/'ra_exp').glob('*.py'):
    s=p.read_text(encoding='utf-8').replace('sc_exp','ra_exp').replace('sc_coreset','ra_coreset').replace('SC_PARENT','RA_PARENT')
    p.write_text(s,encoding='utf-8',newline='\n')
for name in ['train.py','evaluate.py']:
    p=S/'coreset_exp'/name;s=p.read_text(encoding='utf-8').replace('sc_exp','ra_exp').replace('sc_coreset','ra_coreset').replace('SC_PARENT','RA_PARENT')
    p.write_text(s,encoding='utf-8',newline='\n')
shutil.copy2(R/'attachment/ra_coreset_execution_plan/selector_reference.py',S/'ra_exp/selector_reference.py')
shutil.copy2(R/'attachment/ra_coreset_execution_plan/test_reference.py',S/'ra_exp/test_reference.py')
p=S/'ra_exp/test_reference.py';p.write_text(p.read_text().replace('from selector_reference import','from ra_exp.selector_reference import'),newline='\n')
# Remove obsolete SC selection implementation/tests from this new workspace only.
for name in ['selection.py','test_selection.py']:(S/'ra_exp'/name).unlink()
p=S/'ra_exp/storage.py';s=p.read_text().replace('from .selection import repair','from .selector_reference import select_anchor_repair')
s=s.replace("report=repair(ids,binding,[Path(p).name for p in s0],D)","names,report=select_anchor_repair(ids,[Path(p).name for p in s0],D,seed=SEED)")
s=s.replace("        names=report['selected_ids']\n",'')
p.write_text(s,newline='\n')
for name in ['env.sh','pipeline.py','final_process_audit.py','monitor_snapshot.py','package_light.py','supervise.sh','launch_once.py']:
    s=(B/name).read_text(encoding='utf-8').replace('SCv2_Coreset_GOAL_seed42_20261018','RA_Coreset_GOAL_seed42_20261018').replace('sc_exp','ra_exp').replace('sc_coreset','ra_coreset').replace('SC_PARENT','RA_PARENT')
    if name=='env.sh':
        s='\n'.join(x for x in s.splitlines() if 'SCV2_OFFLINE' not in x)+'\n'
    (R/name).write_text(s,encoding='utf-8',newline='\n')
print('Built isolated RA source; core changes only namespace/arm/environment identifiers')
