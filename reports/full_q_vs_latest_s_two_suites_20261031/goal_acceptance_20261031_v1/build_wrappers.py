"""Build external GOAL-only CPU acceptance; no production source changes."""
from pathlib import Path
import ast, json, hashlib, difflib
A=Path(__file__).resolve().parent;R=A.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
source=(R/'final_readonly.py').read_text(encoding='utf8')
values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CHECK_SCRIPT','SCRIPTS']}
scripts={}
for case,original in values['SCRIPTS'].items():
 if not case.startswith('GOAL_'):continue
 s=replace(original,"verify();assert (Q.parent/'controller.exit').read_text()=='0'","verify();assert js(Q.parent/'GOAL_EVALUATION_COMPLETE.json')['episodes']==600\nos.nice(10)")
 s=replace(s,"assert int(p.name)==os.getpid() or str(Q.parent) not in cmd,('Residual',p.name)","assert int(p.name)==os.getpid() or not any(str(Q.parent/('GOAL_'+str(seed)))+'/' in cmd for seed in (41,42,43)),('Residual GOAL',p.name)")
 compile(s,case,'exec');scripts[case]=s
 (A/(case+'_cpu_source.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original/'+case,tofile='goal_only/'+case)),encoding='utf8')
aggregate='''
import subprocess,json,time
def run(script):
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
def goal_light(v):
 return {n:x for n,x in v['light'].items() if any(n.startswith(case+'/') for case in SCRIPTS)}
def verify(v):
 assert not v['canceled_mechanism_processes']
 assert not any('/GOAL_' in p['cmd'] for p in v['processes'])
 marker=json.loads(v['light']['GOAL_EVALUATION_COMPLETE.json']['text']);assert marker['episodes']==600
 for case in SCRIPTS:
  assert v['light'][case+'/train_Q.exit']['text'].strip()==v['light'][case+'/evaluate.exit']['text'].strip()=='0'
  assert len(json.loads(v['light'][case+'/OUTCOMES.json']['text']))==200
 return marker
v=run(CHECK_SCRIPT);marker=verify(v);before=goal_light(v);results={};diagnostics={}
for case,script in SCRIPTS.items():
 r=run(script)
 assert r['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and r['contract_sha256']==v['children'][case]
 for k,rows in r.pop('diagnostics').items():diagnostics[case+'_'+k]=rows
 for n,x in r.pop('light').items():
  if case+'/'+n in v['light']:assert x['sha256']==v['light'][case+'/'+n]['sha256']
 results[case]=r
after=run(CHECK_SCRIPT);assert verify(after)==marker
assert after['contract_sha256']==v['contract_sha256'] and goal_light(after)==before
v.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',scope='GOAL_ONLY_OBJECT_CONTINUES',cases=results,diagnostics=diagnostics,goal_acceptance_boundary=marker,goal_hashes_before={n:x['sha256'] for n,x in before.items()},goal_hashes_after={n:x['sha256'] for n,x in goal_light(after).items()},post_audit_observed=after['observed'],post_audit_processes=after['processes'],environment_calls=0,optimizer_updates=0)
print(json.dumps(v))
'''
(A/'final_readonly.py').write_text('CHECK_SCRIPT='+repr(values['CHECK_SCRIPT'])+'\nSCRIPTS='+repr(scripts)+'\n'+aggregate,encoding='utf8')
s=(R/'audit_final_local.py').read_text(encoding='utf8')
s=replace(s,'R=Path(__file__).resolve().parent;D=Path(sys.argv[1])','R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])')
start=s.index("a=js(D/'LOCAL_AUDIT.json');")
end=s.index('results={}',start)
s=s[:start]+'''a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert v['scope']=='GOAL_ONLY_OBJECT_CONTINUES' and v['goal_hashes_before']==v['goal_hashes_after']
assert v['goal_acceptance_boundary']['episodes']==600 and not any('/GOAL_' in p['cmd'] for p in v['processes']+v['post_audit_processes'])
for case in ['GOAL_41','GOAL_42','GOAL_43']:
 c=a['counts'][case]
 assert (c['epochs'],c['accepted_updates'],c['physical_optimizer_calls'],c['own_isolated_optimizer_calls'],c['own_extra_autograd_calls'],c['training_collection_completed_episodes'],c['formal_episodes'])==(400,76100,76100,91,41,40,200)
assert v['optimizer_updates']==v['environment_calls']==0
'''+s[end:]
s=replace(s,"['GOAL_41', 'GOAL_42', 'GOAL_43', 'OBJECT_41', 'OBJECT_42', 'OBJECT_43']","['GOAL_41', 'GOAL_42', 'GOAL_43']")
for a,b in [("status='PASS_COMPLETE_LOCAL_ACCEPTANCE'","status='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE',scope='GOAL_ONLY_OBJECT_PENDING'"),('epochs=2400,main_updates=544800,physical_training_optimizer_calls=544800,isolated_optimizer_updates=546,formal_extra_autograd=216,admission_extra_autograd=246,training_collection_episodes=240,formal_episodes=1200,vectors=120,cells=60','epochs=1200,main_updates=228300,physical_training_optimizer_calls=228300,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123,training_collection_episodes=120,formal_episodes=600,vectors=60,cells=30')]:s=replace(s,a,b)
(A/'audit_final_local.py').write_text(s,encoding='utf8')
s=(R/'compare_primary_s.py').read_text(encoding='utf8')
for a,b in [('R=Path(__file__).resolve().parent;D=Path(sys.argv[1])','R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])'),("a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==1200","a['status']=='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE' and a['formal_episodes']==600"),("a['main_updates']==544800 and a['isolated_optimizer_updates']==546","a['main_updates']==228300 and a['isolated_optimizer_updates']==273"),("for suite,ref in binding['sources'].items():","for suite,ref in [('GOAL',binding['sources']['GOAL'])]:"),('PASS_NEW_Q_EXISTING_S_1200_PAIRED_EPISODES_EQUAL_ACCEPTED_UPDATES','PASS_GOAL_Q_EXISTING_S_600_PAIRED_EPISODES_EQUAL_ACCEPTED_UPDATES'),('paired_episodes=1200','paired_episodes=600'),('两个套件各三seed，1200回合','仅GOAL套件三seed，600回合'),('两者各240次训练采集','两者各120次训练采集')]:s=replace(s,a,b)
(A/'compare_primary_s.py').write_text(s,encoding='utf8')
for p in A.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
(A/'WRAPPER_REVIEW.json').write_text(json.dumps(dict(status='STATIC_ONLY_PENDING_ACTUAL_CPU',scope='GOAL only; OBJECT continues',completion='GOAL_EVALUATION_COMPLETE and own train/evaluate exit0 and no GOAL process',source_hashes={n:sha(R/n) for n in ['final_readonly.py','audit_final_local.py','compare_primary_s.py']},files={p.name:sha(p) for p in A.glob('*.py')},scientific_mutations=0,optimizer_calls=0,environment_calls=0),indent=2)+'\n',encoding='utf8')
print('STATIC_ONLY wrappers compiled')
