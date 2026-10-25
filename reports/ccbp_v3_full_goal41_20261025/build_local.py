"""Derive a NEW local C/R/G experiment. Never executes remote jobs."""
from pathlib import Path
import ast, json, hashlib
R=Path(__file__).resolve().parent;P=R.parent/'20261024_ccbp_v2_full'
root='/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1'
old='/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02'

def save(n,s):
    if n.endswith('.py'):ast.parse(s)
    (R/n).write_text(s,encoding='utf8')
def read(n):return (P/n).read_text(encoding='utf8')

save('legacy_losses.py',read('legacy_losses.py'))
save('directional_prototype.py',(R/'uploaded_reviewed_text/losses_ccbp_v3.py').read_text(encoding='utf8'))
save('runtime.py',read('runtime.py'))
save('layout.py',"""from pathlib import Path
ARMS='CRG'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/CCBP_V3_FULL_GOAL41_20261025_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/CCBP_V3_FULL_GOAL41_20261025_v1')
OUTPUT={a:(ROOT_PRIVATE if a in 'CR' else DATA_PRIVATE)/a for a in ARMS}
# Conservative proposed bounds, must pass fresh remote preflight before dispatch.
VOLUME_BOUND={'/root':int(1.10*1024**3),'/data':int(1.40*1024**3)}
POP={6:997,7:1388,8:1848,9:3365}
VALID={6:917,7:1286,8:1723,9:3218}
OLD={6:0,7:477,8:899,9:1329}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return (ROOT_PRIVATE if (arm,stage)==('C',6) else DATA_PRIVATE)/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return DATA_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
""")
s=read('updates.py').replace('from losses import aux','from losses import aux,reference_r')
s=s.replace('measure=False,legacy=False):','measure=False,legacy=False,anchor_strength=1.,reference=False):')
s=s.replace('ex=extra;rr=extra;','ex=extra;rr=extra;an=extra;')
s=s.replace('extra,st,ex,rr=aux(arm,q,x[\'next_action\'][eligible],teacher,scale,w)',"extra,st,ex,rr,an=aux(arm,q,x['next_action'][eligible],teacher,scale,w,anchor_strength)")
s=s.replace("legacy_aux(arm,q,", "legacy_aux('C',q,")
s=s.replace("if legacy:extra,_=legacy_aux('C',q,x['next_action'][eligible],teacher,scale)","if legacy:extra,_=legacy_aux('C',q,x['next_action'][eligible],teacher,scale)\n   if reference:extra=reference_r(q,x['next_action'][eligible],teacher,scale,w)")
s=s.replace("for name,term in [('bc',bc),('ifa_weighted',.1*ifa),('exec_weighted',.1*ex),('route_weighted',.01*rr)]:", "terms=[('bc',bc),('ifa_weighted',.1*ifa),('exec_weighted',.1*ex),('route_weighted',.01*rr)]\n  if arm=='G':terms.append(('anchor_weighted',.1*an))\n  for name,term in terms:")
s=s.replace("grad['extra_autograd_calls']=4", "grad['extra_autograd_calls']=len(terms)")
s=s.replace('route_loss=float(rr.detach()),','route_loss=float(rr.detach()),anchor_loss=float(an.detach()),')
save('updates.py',s)
s=read('train.py')
# Remove obsolete B6 continuation entirely; any future recovery requires a new version.
s=s.replace("resume=(arm=='B' and (R/'REPAIR_CONTINUATION.json').exists());assert resume or not (dest/'recovery.pth').exists()", "assert not (dest/'recovery.pth').exists()")
start=s.index('  if resume:');end=s.index('  for stage in range(first,10):',start)
s=s[:start]+s[end:]
s=s.replace("if arm in 'BC':", "if arm=='C':")
needle="restore(a,one);update(a,batch,arm,cache,scale,weights,info,stage,legacy=True);parity=equal(two,capture(a));assert parity;calls+=1"
s=s.replace(needle,needle+"\n  elif arm=='R':\n   restore(a,one);update(a,batch,arm,cache,scale,weights,info,stage,reference=True);parity=equal(two,capture(a));assert parity;calls+=1\n  else:\n   restore(a,one);update(a,batch,arm,cache,scale,weights,info,stage,anchor_strength=0.);off=capture(a)\n   restore(a,one);update(a,batch,arm,cache,scale,weights,info,stage,legacy=True);parity=equal(off,capture(a));assert parity;calls+=2")
s=s.replace("'exec_loss','route_loss','valid'", "'exec_loss','route_loss','anchor_loss','valid'")
save('train.py',s)
s=read('evaluate.py').replace('800','600').replace('vectors=80,cells=40','vectors=60,cells=30').replace("load('B',6", "load('C',6")
save('evaluate.py',s)
s=read('controller.py').replace('CCBP_V2_FULL_GOAL41_20261024','CCBP_V3_FULL_GOAL41_20261025').replace("'BCNV'","'CRG'")
save('controller.py',s)
s=read('check_readonly.py').replace(old,root)
save('check_readonly.py',s)
s=read('audit_progress.py').replace("'BCNV'","'CRG'")
s=s.replace("(4 if t==6 else 5 if arm in 'BC' else 4)","(4 if t==6 else 6 if arm=='G' else 5)")
s=s.replace("if t==6 or arm in 'BC':assert a['baseline_or_legacy_update_exact']","assert a['baseline_or_legacy_update_exact']")
start=s.index("if 'REPAIR_CONTINUATION.json' in light");end=s.index('if not failure',start)
s=s[:start]+s[end:]
save('audit_progress.py',s)
save('audit_evaluation_progress.py',read('audit_evaluation_progress.py').replace('[BCNV]','[CRG]').replace('(80-len(completed))','(60-len(completed))'))
save('remote.py',read('remote.py'))
s=read('publish.py').replace('ccbp_v2_full_goal41_20261024','ccbp_v3_full_goal41_20261025').replace('CCBP-v2 finite GOAL41 four-arm','CCBP-v3 finite GOAL41 three-arm')
save('publish.py',s)
save('capacity_readonly.py',read('capacity_readonly.py').replace('CCBP_V2_FULL_GOAL41_20261024_v1','CCBP_V3_FULL_GOAL41_20261025_v1').replace('1.07*','1.10*').replace('1.88*','1.40*'))
receipt={n.name:hashlib.sha256(n.read_bytes()).hexdigest() for n in R.glob('*.py')}
(R/'LOCAL_DERIVATION_HASHES.json').write_text(json.dumps(dict(status='LOCAL_DERIVATION_ONLY_NOT_DEPLOYED',source_directory=str(P),files=receipt),indent=2)+'\n',encoding='utf8')
print('New local derivation AST checked; zero remote commands, updates, or environments.')
