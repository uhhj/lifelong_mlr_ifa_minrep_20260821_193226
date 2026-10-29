"""Local composition only. Never launches or reruns any previous experiment."""
from pathlib import Path
import ast, json, hashlib, difflib, sys
R=Path(__file__).resolve().parent
N=R.parent/'20261028_support_gate_full_s_object'
G=R.parent/'20261026_ccbp_v4_s_goal42_43'
G41=R.parent/'20261025_ccbp_v4_full'
NAME='SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1'
OLD='SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1'
def read(p):return p.read_text(encoding='utf8')
def put(p,s):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
def dump(p,x):put(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def adapt(s):
 for a,b in [(OLD,NAME),('105500','76100'),('195300','142200'),('89800','66100'),('316500','228300'),('585900','426600'),('269400','198300'),('9100','6700'),('27300','20100')]:s=s.replace(a,b)
 return s
if len(sys.argv)==1:
 s=read(G/'preflight_readonly.py').replace('CCBP_V4_S_GOAL42_43_20261026_v1',NAME).replace('for seed in (42,43):','for seed in (41,42,43):').replace("('/data',4*1024**3","('/data',8*1024**3")
 s=s.replace("assert int(cfg['seed'])==seed and int(ec['seed'])==seed", "assert int(cfg['seed'])==seed and int(ec['seed'])==seed\n assert ec['benchmark_name'].lower()=='libero_goal'")
 s=s.replace('parents={}',"completed=Path('/data/mlr_ifa/experiment_workspaces/"+OLD+"')\nassert (completed/'engineering_copy_20261029_v1/controller.exit').read_text().strip()=='0'\nparents={}")
 s=s.replace('(str(Q) in cmd or str(P) in cmd)',"(str(Q) in cmd or str(P) in cmd or str(completed) in cmd or 'SUPPORT_GATE_FORKS_20261027_v1' in cmd)")
 put(R/'preflight_readonly.py',s);ast.parse(s,feature_version=(3,8))
 print('PREFLIGHT_WRAPPER_ONLY');sys.exit()
prepath=Path(sys.argv[1])/'EVIDENCE.json';pre=json.loads(read(prepath))
assert pre['status']=='PASS_PREDEPLOY_READONLY'
dump(R/'PREFLIGHT_LOCAL_AUDIT.json',dict(status=pre['status'],observed=pre['observed'],raw_path=str(prepath),sha256=sha(prepath),capacity=pre['capacity'],hardware=pre['hardware'],new_scientific_calls=0))
bundle={};review={}
for seed in (41,42,43):
 parent=pre['parents'][str(seed)];source=G41 if seed==41 else G/f'seed{seed}'
 mapping={}
 # Keep the GOAL data, initialization, collection and evaluation implementation.
 for n in ['layout.py','runtime.py','legacy_losses.py','losses.py','memory.py','initialization.py','collect.py','evaluate.py','synthetic_checks.py']:
  mapping[n]=read((G/'seed42' if seed==41 and n in ['initialization.py','collect.py','evaluate.py'] else source)/n)
 # GOAL41's old multiarm directory is only a source, never an execution target.
 s=mapping['layout.py']
 if seed==41:
  s=s.replace("ARMS='DOS'", "ARMS='S'\nSEED=41").replace("OUTPUT={a:DATA_PRIVATE/a for a in ARMS}","OUTPUT={a:DATA_PRIVATE/('seed%d'%SEED)/a for a in ARMS}")
 s=s.replace('CCBP_V4_FULL_GOAL41_20261025_v1',NAME).replace('CCBP_V4_S_GOAL42_43_20261026_v1',NAME).replace("'/data':4*1024**3","'/data':8*1024**3")
 mapping['layout.py']=s
 for key,field in [('POP','population'),('VALID','valid'),('OLD','old_valid')]:
  line=next(x for x in s.splitlines() if x.startswith(key+'='))
  assert ast.literal_eval(line.split('=',1)[1])=={int(k):v[field] for k,v in parent['stages'].items()}
 assert repr(parent['output']) in mapping['runtime.py'] and repr(parent['source']) in mapping['runtime.py']
 # Frozen, accepted gated method. Only suite budgets differ in train.py.
 for n in ['train.py','updates.py','legacy_updates.py','support_memory.py','fork_tools.py','gate_admission.py']:
  mapping[n]=adapt(read(N/f'seed{seed}'/n))
 mapping['legacy_memory.py']=mapping['memory.py']
 mapping['candidate.py']=read(N/'engineering_copy_20261029_v1/candidate.py')
 # One stage-start oracle comparison, with no optimizer/environment calls added.
 mapping['runtime.py']+='''\n# Independent snapshots: CPU inputs retain clone; CUDA transfer already owns storage.
from candidate import cpu_snapshot
_reference_cpu=cpu
_reference_capture=capture
_copy_checked={}
cpu=cpu_snapshot
def capture(a):
 result=_reference_capture(a)
 key=str(a.step)
 if a.step in [0,10000,23900,42400] and a.scheduler.last_epoch==0 and key not in _copy_checked:
  global cpu
  try:
   cpu=_reference_cpu;oracle=_reference_capture(a)
  finally:cpu=cpu_snapshot
  assert equal(result,oracle),'GOAL actual stage initial full state copy mismatch'
  _copy_checked[key]=dict(all_state_exact=True,optimizer_calls_added=0,environments_added=0,time=time.time())
  write(R/'COPY_EQ.json',dict(status='PASS_OBSERVED_STAGE_STARTS',checked=_copy_checked,candidate_sha256=sha(R/'candidate.py')))
 return result
'''
 for n,s in mapping.items():
  ast.parse(s,feature_version=(3,8))
  ref=(G/'seed42'/n) if seed==41 and n in ['initialization.py','collect.py','evaluate.py'] else source/n if (source/n).exists() else N/f'seed{seed}'/n
  original=read(ref) if ref.exists() else ''
  review[f'seed{seed}/{n}']=dict(reference=str(ref),source_sha256=hashlib.sha256(original.encode()).hexdigest(),new_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==original)
  if s!=original:put(R/'review'/f'seed{seed}_{n}.diff',''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(ref),tofile=f'new/seed{seed}/{n}')))
 for n in ['initialization.py','collect.py','evaluate.py','memory.py','losses.py','legacy_losses.py','synthetic_checks.py']:assert mapping[n]==read((G/'seed42' if seed==41 and n in ['initialization.py','collect.py','evaluate.py'] else source)/n)
 mapping['INITIAL_SPLIT.json']=json.dumps(parent['initial_split'],indent=2)+'\n'
 mapping['PARENT_BINDING.json']=json.dumps(parent,indent=2)+'\n'
 mapping['PROTOCOL.md']=read(R/'PROTOCOL.md')
 for n,s in mapping.items():put(R/f'seed{seed}'/n,s);bundle[f'seed{seed}/{n}']=s
put(R/'controller.py',adapt(read(N/'controller.py')))
for n in ['controller.py','PROTOCOL.md']:bundle[n]=read(R/n)
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(prepath),seed41_contract=pre['seed41_contract'],capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
dump(R/'DEPLOY_TEXT_BUNDLE.json',bundle)
dump(R/'UPLOAD_REVIEW.json',dict(status='REVIEWED_TEXT_ONLY_PENDING_REMOTE_IMPORT_GPU_ADMISSION',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},source_review=review,private_payloads_included=False))
put(R/'deploy_remote.py',adapt(read(N/'deploy_remote.py')))
for n in ['check_readonly.py','audit_progress.py']:
 s=adapt(read(N/n))
 if n=='audit_progress.py':
  s=s.replace("own=ownauto=0", "copycheck=js(prefix+'COPY_EQ.json',{})\n if h:\n  for st in range(6,h[-1]['stage']+1):\n   start=sum(p['stages'][str(t)]['batches']*100 for t in range(6,st))\n   assert copycheck['checked'][str(start)]['all_state_exact']\n own=ownauto=0")
 put(R/n,s)
s=read(N/'publish.py').replace('reports/support_gate_full_s_object_20261028','reports/support_gate_full_s_goal_20261029').replace('Latest complete support-gated S OBJECT three seeds','Latest complete support-gated S GOAL three seeds')
s=s.replace("if p.relative_to(R).as_posix() == 'preflight_readonly_20261027_132642/EVIDENCE.json':\n            continue", "if p.name == 'EVIDENCE.json' and p.parent.name.startswith('preflight_readonly_'):\n            pre_audit=json.loads((R/'PREFLIGHT_LOCAL_AUDIT.json').read_text(encoding='utf8'))\n            assert hashlib.sha256(p.read_bytes()).hexdigest()==pre_audit['sha256']\n            continue")
put(R/'publish.py',s)
print(json.dumps(dict(status='BUILT_NOT_DISPATCHED',files=len(bundle),accepted=228300,physical=426600)))
