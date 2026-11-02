"""Compose external recovery only; frozen science/prior engineers unchanged."""
from pathlib import Path
import ast,json,hashlib,difflib
R=Path(__file__).resolve().parent.parent;E=Path(__file__).resolve().parent;PREV=R/'engineering_resume_3090_20261102_v1';CHECK=R/'engineering_checks_3090_20261102_v1'
s=(PREV/'resume_current.py').read_text(encoding='utf8');original=s
pairs=[("len(cp['history'])==260","len(cp['history'])==289"),('len(interrupted_gates)==4167','len(interrupted_gates)==5079'),('interrupted_gates[:4160]','interrupted_gates[:5059]'),('range(60 if stage==8 else 0,100)','range(89 if stage==8 else 0,100)'),('stage==8 and ep==60 and bi<70','stage==8 and ep==89 and bi<200'),('interrupted_gates[4160+bi//10]','interrupted_gates[5059+bi//10]')]
for a,b in pairs:assert s.count(a)==1,a;s=s.replace(a,b)
ast.parse(s);(E/'resume_current.py').write_text(s,encoding='utf8');(E/'resume_current.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='prior_resume',tofile='epoch289_resume')),encoding='utf8')
s=(PREV/'train_wrapper.py').read_text(encoding='utf8')
s=s.replace("resume=case=='L005_41'","resume=case=='L005_43'")
s=s.replace("=='PASS_GPU_RECOVERY_AND_PAIRED_BRANCH_REPLAY'","=='PASS_SAME_GPU_RECOVERY_PARTIAL_PREFIX_AND_NEXT_BLOCK'")
s=s.replace("rt.VOLUME_BOUND['/data']+=346429749","rt.VOLUME_BOUND['/data']+=346429749+346445621")
s=s.replace('65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070','6458bc06f8e99766d360faba812f1c56aff2962616b92931a1806663a750f222')
s=s.replace('checked={}','checked=copy.deepcopy(rt.js(E/\'ORIGINAL_BRANCH_COPY_EQ.json\')[\'checked\']) if resume else {}')
ast.parse(s);(E/'core_wrapper.py').write_text(s,encoding='utf8')
# Preserve the verified checks transformation, with new output provenance and original observed stage checks retained.
s=(CHECK/'train_wrapper.py').read_text(encoding='utf8')
s=s.replace("source=(PREV/'train_wrapper.py').read_text()","source=(NEW/'core_wrapper.py').read_text()")
s=s.replace("finite_starts={};device_starts={}","prior_checks=rt.js(NEW/'ORIGINAL_CHECK_EQ.json') if case=='L005_43' else dict(finite_stage_starts={},device_stage_starts={})\nfinite_starts=copy.deepcopy(prior_checks['finite_stage_starts']);device_starts=copy.deepcopy(prior_checks['device_stage_starts'])")
s=s.replace("rt.write(CHECK_E/(case+'_CHECK_EQ.json')","rt.write(NEW/(case+'_CHECK_EQ.json')")
s=s.replace("scope=dict(__file__=str(PREV/'train_wrapper.py'),__name__='__main__',CHECK_E=CHECK_E)","scope=dict(__file__=str(NEW/'core_wrapper.py'),__name__='__main__',CHECK_E=CHECK_E,NEW=NEW)")
body="""import sys,json,hashlib
from pathlib import Path
NEW=Path(__file__).resolve().parent;Q=NEW.parent
c=json.loads((NEW/'CONTRACT.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
assert json.loads((NEW/'VALIDATION.json').read_text())['status']=='PASS_SAME_GPU_RECOVERY_PARTIAL_PREFIX_AND_NEXT_BLOCK'
assert (Q/'engineering_live_parallel_3090_20261102_v1/HOOK_ENABLED.json').exists()
scope=dict(__file__=str(Q/'engineering_checks_3090_20261102_v1/train_wrapper.py'),__name__='__main__',NEW=NEW)
exec(compile(SOURCE,str(NEW/'checks_adapter.py'),'exec'),scope)
""".replace('SOURCE',repr(s))
ast.parse(body);(E/'train_wrapper.py').write_text(body,encoding='utf8');(E/'checks_adapter.py').write_text(s,encoding='utf8')
print('COMPOSED_UNDEPLOYED_RECOVERY_WRAPPERS')
