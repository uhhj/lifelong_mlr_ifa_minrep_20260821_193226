"""Generate external Q CPU validators; static only, never execute scientific work."""
from pathlib import Path
import ast,json,hashlib
R=Path(__file__).resolve().parent;G=R.parent/'20261029_support_gate_full_s_goal'
NAME='FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1';CASES=[s+'_'+str(k) for s in ['GOAL','OBJECT'] for k in [41,42,43]]
def read(p):return p.read_text(encoding='utf8')
def put(n,s):ast.parse(s,feature_version=(3,8));(R/n).write_text(s,encoding='utf8')
scripts={}
for case in CASES:
 p=json.loads(read(R/case/'PARENT_BINDING.json'));s=read(G/'final_seed41_readonly.py')
 line=next(x for x in s.splitlines() if x.startswith('Q=Path('))
 s=s.replace(line,"Q=Path('/data/mlr_ifa/experiment_workspaces/"+NAME+'/'+case+"');O=Path("+repr(p['output'])+");S=Path("+repr(p['source'])+")")
 starts={str(sum(p['stages'][str(t)]['batches']*100 for t in range(6,st))) for st in range(6,10)}
 s=s.replace("{'0','10000','23900','42400'}",repr(starts))
 s=s.replace('MEMORY_S_','MEMORY_Q_').replace('TRAIN_INPUT_S_','TRAIN_INPUT_Q_').replace('single_S_only','single_Q_only')
 s=s.replace("gates=[json.loads(line) for line in (Q/'GATES.jsonl').read_text().splitlines()]\n assert len(gates)==6700","assert not (Q/'GATES.jsonl').exists()")
 start=s.index("  subset=[g for g in gates");end=s.index(' rec=torch.load(',start)
 s=s[:start]+'''  for ep in range(100):
   rr=hist[(stage-6)*100+ep];physical+=nb
   assert rr['physical_optimizer_calls']==physical and rr['physical_calls_observed']==dict(production=physical,admission=4+29*(stage-6))
   assert rr['totals']['anchor_slots']==(4*nb if stage>6 else 0) and rr['gate_blocks']==rr['gate_Q_accepts']==0
  offset+=100*nb
  if stage>6:by_stage[str(stage)]=dict(support=pr['support'],fixed_probes_verified=True)
 assert physical==offset==TOTAL_UPDATES
 production_audit=dict(status='PASS_INDEPENDENT_Q_NO_GATE_RECOMPUTATION',blocks=0,physical_training_optimizer_calls=physical,accepted_updates=offset,by_stage=by_stage)
'''+s[end:]
 s=s.replace('==76100','==TOTAL_UPDATES').replace('==142200','==TOTAL_UPDATES').replace("rec['gate_blocks']==6700","rec['gate_blocks']==0").replace('gate_audit=gate_audit','production_audit=production_audit')
 needle="assert len(task6)==1 and len(collections)==40"
 s=s.replace(needle,needle+"\nsref=js(Q/'S_REFERENCE.json');assert all(sha(x['path'])==x['sha256'] for x in sref.values())\nss=torch.load(sref['stage6_checkpoint']['path'],map_location='cpu')['state_dict'];ss={(k[7:] if k.startswith('module.') else k):ah(v.numpy()) for k,v in ss.items()};assert ss==task6['Q']\npb=js(Q/'PARENT_BINDING.json');assert all(sha(p)==h for p,h in pb['files'].items())")
 put('final_'+case+'_readonly.py',s);scripts[case]=s
agg='''"""Only after natural completion: CPU audits, no optimizer or environments."""
import subprocess,json,time
def run(script):
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
v=run(CHECK_SCRIPT)
assert v['light'].get('controller.exit',{}).get('text','').strip()=='0' and not v['processes'] and not v['canceled_mechanism_processes']
results={};diagnostics={}
for case,script in SCRIPTS.items():
 r=run(script);print('',flush=True)
 assert r['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and r['contract_sha256']==v['children'][case]
 for k,rows in r.pop('diagnostics').items():diagnostics[case+'_'+k]=rows
 for n,x in r.pop('light').items():
  if case+'/'+n in v['light']:assert x['sha256']==v['light'][case+'/'+n]['sha256']
 results[case]=r
after=run(CHECK_SCRIPT)
assert after['contract_sha256']==v['contract_sha256'] and after['light']==v['light'] and not after['processes']
v.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',observed=time.time(),cases=results,diagnostics=diagnostics)
print(json.dumps(v))
'''
put('final_readonly.py','CHECK_SCRIPT='+repr(read(R/'check_readonly.py'))+'\nSCRIPTS='+repr(scripts)+'\n'+agg)
s=read(G/'audit_seed_evaluation.py').replace('([S])','([Q])');put('audit_seed_evaluation.py',s)
s=read(G/'audit_final_local.py')
s=s.replace("a['epochs']==1200","a['epochs']==2400").replace("a['accepted_updates']==228300","a['accepted_updates']==544800").replace("a['own_isolated_optimizer_calls']==273","a['own_isolated_optimizer_calls']==546").replace("a['own_extra_autograd_calls']==123","a['own_extra_autograd_calls']==246").replace("a['physical_training_optimizer_calls']==426600","a['physical_training_optimizer_calls']==544800").replace("a['training_collection_completed_episodes']==120","a['training_collection_completed_episodes']==240").replace("a['formal_episodes']==600","a['formal_episodes']==1200")
s=s.replace('for seed in (41,42,43):', 'for case in '+repr(CASES)+':')
start=s.index(" key=str(seed);");end=s.index(" child=D/",start)
s=s[:start]+" seed=int(case.split('_')[1]);key=case;r=dict(v['cases'][case],light={n[len(case)+1:]:x for n,x in v['light'].items() if n.startswith(case+'/')});p=js(R/case/'PARENT_BINDING.json');progress=a['counts'][case]\n"+s[end:]
s=s.replace("child=D/('final_seed'+key)","child=D/('final_case_'+key)")
s=s.replace("str(child),key],check=True)","str(child),str(seed)],check=True)")
s=s.replace('single_S_only','single_Q_only').replace('train_S.exit','train_Q.exit').replace('TRAIN_S.json','TRAIN_Q.json').replace('GRADIENTS_S.json','GRADIENTS_Q.json').replace('MEMORY_S_','MEMORY_Q_').replace('SHORT_HISTORY_ADMISSION_S','SHORT_HISTORY_ADMISSION_Q').replace("('S',t)","('Q',t)").replace("['strict']['S']","['strict']['Q']").replace("[f'S_{t}']","[f'Q_{t}']").replace("arm=='S'","arm=='Q'")
s=s.replace("fp.stem.startswith(f's{seed}_')","fp.stem.startswith(case+'_')").replace("_,kind,arm,st=fp.stem.split('_')","kind,arm,st=fp.stem[len(case)+1:].split('_')")
s=s.replace(" assert r['gate_audit']['status']=='PASS_INDEPENDENT_RECOMPUTATION' and r['gate_audit']['blocks']==6700 and r['gate_audit']['physical_training_optimizer_calls']==142200", " assert r['production_audit']['status']=='PASS_INDEPENDENT_Q_NO_GATE_RECOMPUTATION' and r['production_audit']['blocks']==0 and r['production_audit']['physical_training_optimizer_calls']==sum(x['batches']*100 for x in p['stages'].values())")
s=s.replace("gate_audit=r['gate_audit']","production_audit=r['production_audit']")
s=s.replace('seeds=results,epochs=1200,main_updates=228300,physical_training_optimizer_calls=426600,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123,training_collection_episodes=120,formal_episodes=600,vectors=60,cells=30','cases=results,epochs=2400,main_updates=544800,physical_training_optimizer_calls=544800,isolated_optimizer_updates=546,formal_extra_autograd=216,admission_extra_autograd=246,training_collection_episodes=240,formal_episodes=1200,vectors=120,cells=60')
s=s.replace('Only latest support-gated S GOAL41/42/43; the canceled mechanism experiment is not resumed or required for acceptance.','Only six new Q; accepted S is read-only; canceled mechanism is not resumed.').replace('Historical methods are read-only references; extra acquisition and double candidates prohibit equal-interaction or equal-compute causal claims.','Q and S match accepted epochs and collection counts, not cache contents or physical optimizer/FLOPs/wall-clock costs.')
put('audit_final_local.py',s)
files=['final_readonly.py','audit_final_local.py','audit_seed_evaluation.py']+['final_'+c+'_readonly.py' for c in CASES]
(R/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(dict(status='STATIC_ONLY_NOT_ACCEPTED',files={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files},actual_final_CPU_calls=0,requires='Natural controller.exit0 and no own processes, then actual 24CP/6recovery/24cache/1200initial-noise verification'),indent=2),encoding='utf8')
print('STATIC_FINAL_WRAPPERS_READY')
