"""External suite-only acceptance adaptation; never deploys or runs science."""
from pathlib import Path
import ast,json,hashlib,difflib,datetime
A=Path(__file__).resolve().parent;R=A.parent;E=R/'engineering_suite_order_20261030_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
src=(E/'final_readonly.py').read_text(encoding='utf8');tree=ast.parse(src)
original=ast.literal_eval(tree.body[0].value);scripts={}
for case,s in original.items():
 if not case.startswith('GOAL_'):continue
 s=replace(s,"verify();assert (Q.parent/'engineering_suite_order_20261030_v1/controller.exit').read_text()=='0'", "verify();assert js(Q.parent/'engineering_suite_order_20261030_v1/GOAL_EVALUATION_COMPLETE.json')['episodes']==600\nos.nice(10)")
 s=replace(s,"assert int(p.name)==os.getpid() or str(Q.parent) not in cmd,('Residual',p.name)","assert int(p.name)==os.getpid() or not any(str(Q.parent/('GOAL_'+str(seed)))+'/' in cmd for seed in (41,42,43)),('Residual GOAL',p.name)")
 compile(s,case,'exec');scripts[case]=s
 (A/(case+'_cpu_source.diff')).write_text(''.join(difflib.unified_diff(original[case].splitlines(True),s.splitlines(True),fromfile='original/'+case,tofile='GOAL_only/'+case)),encoding='utf8')
aggregate='''
import os,json,hashlib,time,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1')
E=Q/'engineering_suite_order_20261030_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
def processes():
 out=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  except OSError:continue
  if int(p.name)!=os.getpid() and str(Q) in cmd:out.append(dict(pid=int(p.name),cmd=cmd))
 assert not any(any(str(Q/('GOAL_'+str(seed)))+'/' in x['cmd'] for seed in (41,42,43)) for x in out)
 return out
def verify():
 assert not (Q/'controller.exit').exists()
 ec=js(E/'CONTRACT.json');c=js(Q/'CONTRACT.json')
 assert all(sha(p)==h for p,h in ec['files'].items()) and all(sha(p)==h for p,h in c['files'].items())
 marker=js(E/'GOAL_EVALUATION_COMPLETE.json');assert marker['episodes']==600 and marker['next_phase']=='OBJECT_training'
 for case in SCRIPTS:
  assert (Q/case/'train_C.exit').read_text()==(Q/case/'evaluate.exit').read_text()=='0'
  assert len(js(Q/case/'OUTCOMES.json'))==200
 return c,ec
def goal_hashes():
 return {str(p.relative_to(Q)):sha(p) for case in SCRIPTS for p in (Q/case).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.exit')}
c,ec=verify();before=goal_hashes();process_before=processes();results={};diagnostics={};light={}
for case,script in SCRIPTS.items():
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True)
 assert p.returncode==0,(case,p.stderr)
 v=json.loads(p.stdout)
 for k,rows in v.pop('diagnostics').items():diagnostics[case+'_'+k]=rows
 for k,x in v.pop('light').items():light[case+'/'+k]=x
 results[case]=v
verify();assert before==goal_hashes(),'GOAL evidence changed during CPU audit';process_after=processes()
for p in [Q/'STATUS.json']+list(E.glob('*')):
 if p.is_file() and p.suffix in ('.json','.exit') and p.name!='CONTRACT.json':light[str(p.relative_to(Q))]=dict(text=p.read_text(),sha256=sha(p))
print(json.dumps(dict(status='PASS_GOAL_REMOTE_CPU_PENDING_LOCAL_AUDIT',scope='GOAL_ONLY_OBJECT_CONTINUES',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),children=c['children'],source_files=len(c['files']),engineering_contract=sha(E/'CONTRACT.json'),engineering_files=len(ec['files']),light=light,processes=process_after,processes_before=process_before,goal_processes=[],goal_hashes_before=before,goal_hashes_after=goal_hashes(),cases=results,diagnostics=diagnostics,environment_calls=0,optimizer_updates=0)))
'''
(A/'final_readonly.py').write_text('SCRIPTS='+repr(scripts)+'\n'+aggregate,encoding='utf8')
s=(E/'audit_progress.py').read_text(encoding='utf8')
s=replace(s,"cases=['GOAL_41', 'GOAL_42', 'GOAL_43', 'OBJECT_41', 'OBJECT_42', 'OBJECT_43']","cases=['GOAL_41', 'GOAL_42', 'GOAL_43']")
s=replace(s,"if not failure and L.get('engineering_suite_order_20261030_v1/controller.exit')!='0' and not v.get('processes'):","if not failure and L.get('engineering_suite_order_20261030_v1/GOAL_EVALUATION_COMPLETE.json',{}).get('episodes')!=600:")
s=replace(s,"if any(counts['OBJECT_'+str(s)]['epochs'] for s in [41,42,43]):","if True:")
s=replace(s,"completion_source='engineering_suite_order_20261030_v1/controller.exit'","completion_source='engineering_suite_order_20261030_v1/GOAL_EVALUATION_COMPLETE.json'")
(A/'audit_progress.py').write_text(s,encoding='utf8')
s=(E/'audit_final_local.py').read_text(encoding='utf8')
for a,b in [
 ("v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and not v['processes']","v['status']=='PASS_GOAL_REMOTE_CPU_PENDING_LOCAL_AUDIT' and not v['goal_processes']"),
 ("a['formal_episodes']==1200 and a['isolated_optimizer_calls']==546","a['formal_episodes']==600 and a['isolated_optimizer_calls']==273"),
 ("==4200 and sum(x['steps'] for x in a['counts'].values())==1012500","==2100 and sum(x['steps'] for x in a['counts'].values())==426600"),
 ("assert L['engineering_suite_order_20261030_v1/controller.exit']=='0'","assert L['engineering_suite_order_20261030_v1/GOAL_EVALUATION_COMPLETE.json']['episodes']==600\nassert set(v['cases'])=={'GOAL_41','GOAL_42','GOAL_43'} and v['goal_hashes_before']==v['goal_hashes_after']"),
 ("status='PASS_COMPLETE_LOCAL_ACCEPTANCE'","status='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE',scope='GOAL_ONLY_OBJECT_NOT_ACCEPTED'"),
 ("epochs=4200,updates=1012500,isolated_updates=546,formal_extra_autograd=288,admission_extra_autograd=72","epochs=2100,updates=426600,isolated_updates=273,formal_extra_autograd=144,admission_extra_autograd=36"),
 ("episodes=1200,vectors=120,cells=60","episodes=600,vectors=60,cells=30"),
 ("strict_CPs=24,complete_recoveries=6,teacher_caches=18","strict_CPs=12,complete_recoveries=3,teacher_caches=9")]:s=replace(s,a,b)
(A/'audit_final_local.py').write_text(s,encoding='utf8')
s=(R/'compare_primary_s.py').read_text(encoding='utf8')
for a,b in [
 ("R=Path(__file__).resolve().parent;D=Path(sys.argv[1])","R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])"),
 ("a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['evaluation']['episodes']==1200","a['status']=='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE' and a['evaluation']['episodes']==600"),
 ("a['training']['updates']==1012500 and a['training']['isolated_updates']==546","a['training']['updates']==426600 and a['training']['isolated_updates']==273"),
 ("for suite,ref in binding['sources'].items():","for suite,ref in [('GOAL',binding['sources']['GOAL'])]:"),
 ("status='PASS_NEW_C_EXISTING_S_1200_PAIRED_EPISODES_EQUAL_STAGE_OPTIMIZER_BUDGET'","status='PASS_GOAL_NEW_C_EXISTING_S_600_PAIRED_EPISODES_EQUAL_STAGE_OPTIMIZER_BUDGET'"),
 ("paired_episodes=1200","paired_episodes=600"),
 ("两个套件各三seed，1200回合","仅GOAL套件三seed，600回合"),
 ("已有S的240次训练采集","GOAL已有S的120次训练采集")]:s=replace(s,a,b)
(A/'compare_primary_s.py').write_text(s,encoding='utf8')
files={}
for p in A.glob('*.py'):
 compile(p.read_text(encoding='utf8'),str(p),'exec');files[p.name]=sha(p)
review=dict(status='STATIC_ONLY_PENDING_ACTUAL_GOAL_CPU_ACCEPTANCE',time=datetime.datetime.now().astimezone().isoformat(),authorization='User: 先把GOAL验收了; OBJECT remains running',scope='GOAL_41/42/43 only',completion_source='GOAL_EVALUATION_COMPLETE.json plus each train/evaluate exit0 and no GOAL processes',source_hashes={n:sha(E/n) for n in ['final_readonly.py','audit_progress.py','audit_final_local.py']},files=files,remote_mutations=0,environment_calls=0,optimizer_calls=0,checks='Original per-case CPU contents unchanged except suite completion and residual-process scope, single-core low priority CPU; pre/post GOAL hashes exact',whole_experiment_acceptance=False)
(A/'WRAPPER_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(review,ensure_ascii=False))
