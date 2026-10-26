"""Deterministic seed-only adaptation of accepted engineering02 S; no dispatch."""
from pathlib import Path
import ast,json,hashlib,difflib,sys
R=Path(__file__).resolve().parent;P=R.parent/'20261025_ccbp_v4_full'
Q='/data/mlr_ifa/experiment_workspaces/CCBP_V4_S_GOAL42_43_20261026_v1'
OUT='/data/mlr_ifa/experiments/CCBP_V4_S_GOAL42_43_20261026_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
pre=read(Path(sys.argv[1])/'EVIDENCE.json');assert pre['status']=='PASS_PREDEPLOY_READONLY'
assert read(P/'RUN_STATUS.json')['status']=='COMPLETE_ACCEPTED_NO_AUTOMATIC_EXPANSION'
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
review={};bundle={};finalscripts={}
names=['runtime.py','legacy_losses.py','losses.py','memory.py','updates.py','initialization.py','train.py','collect.py','evaluate.py','synthetic_checks.py']
for seed in (42,43):
 parent=pre['parents'][str(seed)];d=R/f'seed{seed}';mapping={}
 layout=(P/'layout.py').read_text(encoding='utf8').replace("ARMS='DOS'","ARMS='S'\nSEED="+str(seed)).replace("NAME='CCBP_V4_FULL_GOAL41_20261025_v1'","NAME='CCBP_V4_S_GOAL42_43_20261026_v1'")
 layout=layout.replace("DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME","DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME")
 layout=layout.replace("OUTPUT={a:DATA_PRIVATE/a for a in ARMS}","OUTPUT={a:DATA_PRIVATE/('seed%d'%SEED)/a for a in ARMS}")
 for key,field in [('POP','population'),('VALID','valid'),('OLD','old_valid')]:
  line=next(x for x in layout.splitlines() if x.startswith(key+'='));layout=layout.replace(line,key+'='+repr({int(k):v[field] for k,v in parent['stages'].items()}))
 mapping['layout.py']=layout
 for n in names:
  original=(P/n).read_text(encoding='utf8');s=original
  if n=='runtime.py':
   s=s.replace('ALIGN_GOAL_seed41_20261022',f'ALIGN_GOAL_seed{seed}_20261022').replace('/GOAL_41/source',f'/GOAL_{seed}/source').replace("CORESET_SEED='41'",f"CORESET_SEED='{seed}'")
  if n=='initialization.py':
   begin=s.index(" if arm!='D':");end=s.index(" write(R/('INITIAL_'",begin);s=s[:begin]+s[end:]
  if n=='train.py':
   begin=s.index("                ref=actual if arm=='D'");end=s.index("            write(R/('STAGE_",begin)
   s=s[:begin]+"                write(R/('TASK6_PARITY_'+arm+'.json'),dict(historical_ALIGN_equal=equal(actual,baseline),historical_equality_gate=False,actual_sha256=sha(target),historical_ALIGN_sha256=sha(O/'ra_coreset/models/mlrifa_6_model_ep99.pth'),same_seed_new_arm_comparator='NOT_RUN_SINGLE_S'))\n"+s[end:]
  if n in ('collect.py','evaluate.py'):
   s=s.replace('control_seed(41)','control_seed(SEED)').replace('41*1000','SEED*1000').replace('90141','(90100+SEED)').replace('env.seed(141)','env.seed(SEED+100)').replace('env_seed=141','env_seed=SEED+100').replace('41+group','SEED+group')
  if n=='evaluate.py':
   s=s.replace('Exactly 600 paired episodes','Exactly 200 episodes per seed').replace("load('D',6","load('S',6").replace('budget=600','budget=200').replace('len(records)==600','len(records)==200').replace('episodes=600,vectors=60,cells=30','episodes=200,vectors=20,cells=10')
  mapping[n]=s
  review[f'seed{seed}/{n}']=dict(parent_sha256=sha(P/n),new_text_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==original)
  if s!=original:put(R/'review'/f'seed{seed}_{n}.diff',''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='accepted_seed41/'+n,tofile=f'seed{seed}/'+n)))
 mapping['INITIAL_SPLIT.json']=json.dumps(parent['initial_split'],indent=2)+'\n'
 mapping['PARENT_BINDING.json']=json.dumps(parent,indent=2)+'\n'
 mapping['PROTOCOL.md']=(R/'PROTOCOL.md').read_text(encoding='utf8')
 for n,s in mapping.items():
  if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
  put(d/n,s);bundle[f'seed{seed}/{n}']=s
 # Derive CPU-only final wrapper, preserve full source/cache/recovery checks.
 f=(P/'final_readonly.py').read_text(encoding='utf8')
 start=f.index("Q=Path(");end=f.index('\nsys.path.insert',start)
 f=f[:start]+f"Q=Path('{Q}/seed{seed}');O=Path('{parent['output']}');S=Path('{parent['source']}')"+f[end:]
 f=f.replace("assert (Q/'controller.exit').read_text()=='0'","assert (Q.parent/'controller.exit').read_text()=='0'")
 f=f.replace("str(Q) not in cmd","str(Q.parent) not in cmd")
 f=f.replace("(contract['inherited_D_contract'] if arm=='D' else sha(Q/'CONTRACT.json'))","sha(Q/'CONTRACT.json')")
 f=f.replace('1,141)','1,SEED+100)').replace('41000','SEED*1000').replace('90141','(90100+SEED)').replace('==41+r[','==SEED+r[')
 start=f.index("inherited=js(Q/'INHERITED_D_EXIT_OBSERVATION.json')");end=f.index('for r in records:',start)
 f=f[:start]+"for arm in ARMS:\n path_ad=js(Q/('SHORT_HISTORY_ADMISSION_'+arm+'.json'));assert path_ad['status']=='PASS' and path_ad['eight_frame_legacy_exact'] and path_ad['all_state_restored'] and [x['length'] for x in path_ad['lengths']]==list(range(1,9))\nassert len(task6)==1 and len(collections)==40\nrecords=js(Q/'OUTCOMES.json');assert len(records)==200;seen=set()\n"+f[end:]
 line=next(x for x in f.splitlines() if x.startswith(' rr=ref['));f=f.replace(line,'')
 f=f.replace('task6_current_arms_exact=True','task6_current_arms_exact=False,single_S_only=True').replace('reference_C_pairing_exact=True','reference_C_pairing_exact=False').replace('episodes=600,training_collection_episodes=120','episodes=200,training_collection_episodes=40')
 ast.parse(f,feature_version=(3,8));put(R/f'final_seed{seed}_readonly.py',f);finalscripts[str(seed)]=f
bundle['controller.py']=(R/'controller.py').read_text(encoding='utf8')
bundle['PROTOCOL.md']=(R/'PROTOCOL.md').read_text(encoding='utf8')
bundle['PREDEPLOY.json']=json.dumps(dict(observed=pre['observed'],evidence_sha256=sha(Path(sys.argv[1])/'EVIDENCE.json'),seed41_contract=pre['seed41_contract'],capacity=pre['capacity'],hardware=pre['hardware'],source_review=review),indent=2)+'\n'
for n,s in bundle.items():
 if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
put(R/'DEPLOY_TEXT_BUNDLE.json',json.dumps(bundle,ensure_ascii=True,indent=2)+'\n')
put(R/'UPLOAD_REVIEW.json',json.dumps(dict(status='REVIEWED_TEXT_ONLY',files={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in bundle.items()},parent_implementation=review,excludes='credentials/.session_control/private payloads/archives'),indent=2)+'\n')
put(R/'final_readonly.py',"# External CPU audit; no server writes or formal rollouts.\nimport subprocess,json\nSCRIPTS="+ascii(finalscripts)+"\nresults={};diagnostics={}\nfor seed,script in SCRIPTS.items():\n p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True);assert p.returncode==0,(seed,p.stderr)\n v=json.loads(p.stdout)\n for k,rows in v.pop('diagnostics').items():diagnostics['s'+seed+'_'+k]=rows\n results[seed]=v\nprint(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',seeds=results,diagnostics=diagnostics)))\n")
print('Built reviewed S42/S43 sources and CPU final wrappers; no deployment.')
