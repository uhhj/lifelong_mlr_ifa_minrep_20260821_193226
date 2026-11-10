"""Resolve duplicate algorithm class registration before any scientific call."""
from pathlib import Path
import ast,json,hashlib,difflib
R=Path(__file__).resolve().parent;OLD=R/'engineering_environment_20261110_v3';D=R/'engineering_startup_20261110_v4'
assert not D.exists();D.mkdir();sha=lambda t:hashlib.sha256(t.encode()).hexdigest()
old=json.loads((OLD/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'));b=dict(old);gc=json.loads(b['CONTRACT.json'])
changed={}
for case in gc['cases']:
 n='cases/'+case+'/source/coreset_exp/train.py';s=b[n]
 assert s.count('class PaperMLR(MLR):')==1 and s.count('a=PaperMLR(1,c).cuda()')==1
 t=s.replace('class PaperMLR(MLR):','class AlignPaperMLR(MLR):').replace('a=PaperMLR(1,c).cuda()','a=AlignPaperMLR(1,c).cuda()')
 before=ast.parse(s);after=ast.parse(t)
 bc=next(x for x in before.body if isinstance(x,ast.ClassDef) and x.name=='PaperMLR')
 ac=next(x for x in after.body if isinstance(x,ast.ClassDef) and x.name=='AlignPaperMLR')
 bc.name='AlignPaperMLR';assert ast.dump(bc)==ast.dump(ac)
 b[n]=t;gc['code_files'][n]=sha(t)
 cn='private/'+case+'/CONTRACT.json';c=json.loads(b[cn]);c['source_sha256']['coreset_exp/train.py']=sha(t);b[cn]=json.dumps(c,indent=2)+'\n'
 gc['private_files'][case+'/CONTRACT.json']=sha(b[cn])
 changed[case]=dict(before=sha(s),after=sha(t),same_observe_method_AST=True)
 p=D/'source_patches'/case/'train.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf8')
 p.with_suffix('.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile='prepared/train.py',tofile='registration_fix/train.py')),encoding='utf8')
 p=D/'metadata'/case/'CONTRACT.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(b[cn],encoding='utf8')
b['CONTRACT.json']=json.dumps(gc,indent=2)+'\n';contract=sha(b['CONTRACT.json'])
assert all(sha(b[n])==h for n,h in gc['code_files'].items())
assert all(sha(b['private/'+n])==h for n,h in gc['private_files'].items())
for n in ['DEPLOY_TEXT_BUNDLE.json','PREPARED_CONTRACT.json']:(D/n).write_text(json.dumps(b) if n.startswith('DEPLOY') else b['CONTRACT.json'],encoding='utf8')
for n in ['runtime.py','deploy_remote.py','transport.py','launch_after_q1.py']:(D/n).write_text((OLD/n).read_text(encoding='utf8'),encoding='utf8')
for n in ['check_readonly.py','final_readonly.py','record_startup.py']:
 t=(OLD/n).read_text(encoding='utf8').replace(sha(old['CONTRACT.json']),contract).replace('engineering_environment_20261110_v3','engineering_startup_20261110_v4')
 (D/n).write_text(t,encoding='utf8')
runtime=b['runtime.py']
sources={case:b['cases/'+case+'/source/coreset_exp/train.py'] for case in gc['cases']}
pre='exec('+repr(runtime)+')\nSOURCES='+repr(sources)+'\nprevious_finished()\nassert not Q.exists() and not P.exists()\n'
pre+='''results={}
failed=Q.parent/'MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1'
for case in CASES:
 e=env(case);e['CORESET_OUT']=str(P.parent/failed.name/case);e['PYTHONPATH']=str(failed/'cases'/case/'source')
 code="import importlib; scope={'__name__':'coreset_exp.train_import_check','__package__':'coreset_exp'}; exec(compile("+repr(SOURCES[case])+",'reviewed/train.py','exec'),scope); importlib.import_module('coreset_exp.evaluate'); print('IMPORT_ONLY_PASS_NO_MAIN')"
 p=subprocess.run([PY,'-B','-c',code],env=e,cwd=failed/'cases'/case/'source',capture_output=True,text=True)
 assert p.returncode==0,(case,p.stderr[-4000:]);assert 'IMPORT_ONLY_PASS_NO_MAIN' in p.stdout
 results[case]=dict(exit=0,status='PASS_PATCHED_TRAIN_AND_ORIGINAL_EVALUATOR_IMPORT_WITHOUT_MAIN',source_sha256=hashlib.sha256(SOURCES[case].encode()).hexdigest())
previous_finished()
g=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True);assert g.returncode==0 and not g.stdout.strip()
print(json.dumps(dict(status='PASS_SIX_IMPORTS_CONFIG_AND_FRESH_CAPACITY',observed=time.time(),imports=results,capacity=capacity(),optimizer_updates=0,environments=0,model_initializations=0,old_failure_retained=True)))
'''
(D/'preflight_readonly.py').write_text(pre,encoding='utf8')
for p in D.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8))
review=dict(status='PASS_STATIC_CLASS_REGISTRATION_REPAIR_PENDING_ACTUAL_IMPORT',adapted_contract=contract,parent_contract=sha(old['CONTRACT.json']),class_name_changes=changed,science_method_bodies_exact=True,all_other_source_and_private_inputs_exact=True,optimizer_calls=0,environments=0,original_failure_preserved=True)
(D/'REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
print(json.dumps(review))
