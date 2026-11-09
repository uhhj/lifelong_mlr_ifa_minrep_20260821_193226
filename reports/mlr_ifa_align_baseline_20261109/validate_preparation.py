"""Prepared-byte closure and zero-science finite-queue simulations."""
from pathlib import Path
import json,hashlib,ast,tempfile,types,traceback,datetime
R=Path(__file__).resolve().parent
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda b:hashlib.sha256(b).hexdigest()
b=js(R/'DEPLOY_TEXT_BUNDLE.json');c=json.loads(b['CONTRACT.json']);u=js(R/'UPLOAD_REVIEW.json')
assert sha(b['CONTRACT.json'].encode())==u['contract_sha256']
for n,v in u['files_sha256'].items():assert sha(b[n].encode())==v['sha256']
for n,h in c['code_files'].items():assert sha(b[n].encode())==h
for n,h in c['private_files'].items():assert sha(b['private/'+n].encode())==h
for case in c['cases']:
 cc=json.loads(b['private/'+case+'/CONTRACT.json'])
 for n,h in cc['source_sha256'].items():assert sha(b['cases/'+case+'/source/'+n].encode())==h
 for n,h in cc['files'].items():
  prefix='/data/mlr_ifa/experiments/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1/'
  if n.startswith(prefix):assert sha(b['private/'+n[len(prefix):]].encode())==h,n
 for t in range(6,10):
  m=json.loads(b['private/'+case+'/manifests/task%d.json'%t]);assert len(m['rows'])==m['population'] and len({r['relative_path'] for r in m['rows']})==m['population']
 for file in ['model.py','evaluate.py','eval_metric.py']:
  original=Path(js(R/'SOURCE_LOOKUP.json')[case]['coreset_exp/'+file]);expected=original.read_text(encoding='utf-8-sig')
  assert b['cases/'+case+'/source/coreset_exp/'+file]==expected
for n,s in b.items():
 if n.endswith('.py'):ast.parse(s,feature_version=(3,8));compile(s,n,'exec')
# The one-shot deployer must fail before any write when closure is absent.
try:exec(compile((R/'deploy_remote.py').read_text(encoding='utf8'),'deploy','exec'),{'BUNDLE':{}})
except AssertionError as e:assert 'Missing independently accepted' in str(e)
else:raise AssertionError('Deployment gate failed')
def simulate(fail=False):
 with tempfile.TemporaryDirectory(prefix='mlr_align_queue_check_') as folder:
  q=Path(folder)/'code';p=Path(folder)/'private';q.mkdir();p.mkdir();events=[];running=set();completed=[];maximum=[0]
  class Proc:
   def __init__(self,case,kind):self.case=case;self.kind=kind;self.pid=100+len(events)
   def poll(self):return 1 if fail and self.case=='OBJECT_41' else 0
   def wait(self):return self.poll()
  def launch(case,kind):
   if kind=='train':
    assert len(running)<2;running.add(case);maximum[0]=max(maximum[0],len(running))
    if case.startswith('GOAL'):assert len([x for x in completed if x.startswith('OBJECT')])==3
   else:assert not running and sum(x[0]=='train_exit' and x[1].startswith(case.split('_')[0]) for x in events)==3
   events.append((kind,case));return Proc(case,kind)
  def reap(case,kind,proc):
   if kind=='train':running.remove(case)
   else:completed.append(case)
   events.append((kind+'_exit',case));return proc.wait()
  def write(path,x):path.write_text(json.dumps(x),encoding='utf8')
  def fakejs(path):return {'epochs':400,'episodes':200}
  for case in c['cases']:(p/case).mkdir()
  ns=dict(Q=q,P=p,fcntl=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2),time=types.SimpleNamespace(time=lambda:1,sleep=lambda _:None),os=types.SimpleNamespace(getpid=lambda:99),write=write,js=fakejs,verify=lambda:None,acceptance_gate=lambda:None,capacity=lambda:None,processes=lambda _:[],traceback=traceback)
  tree=ast.parse((R/'controller.py').read_text(encoding='utf8'));tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
  exec(compile(tree,'controller_exact_functions','exec'),ns);ns.update(launch=launch,reap=reap)
  if fail:
   try:ns['main']()
   except RuntimeError:pass
   else:raise AssertionError('Failure continued')
   assert (q/'controller.exit').read_text().strip()=='1' and not any(case=='OBJECT_43' for _,case in events) and not running
  else:
   ns['main']();assert (q/'controller.exit').read_text().strip()=='0' and len(completed)==6 and maximum[0]==2
  return dict(failure_injected=fail,events=events,maximum_trainers=maximum[0],new_training=0,new_environments=0)
out=dict(status='PASS_PREPARED_BYTES_STATIC_AND_ZERO_SCIENCE_QUEUE_CHECKS',observed=datetime.datetime.now().astimezone().isoformat(),contract=u['contract_sha256'],source_files=len(c['code_files']),private_metadata_files=len(c['private_files']),all_child_transport_sha_exact=True,original_evaluator_model_and_eval_metric_lf_exact=True,missing_q1_acceptance_stops_before_writes=True,queue_checks=[simulate(),simulate(True)],gpu_admission='NOT_EXECUTED_REQUIRES_POST_Q1_REAL_GPU',new_optimizer=0,new_environments=0)
(R/'PREPARATION_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(out['status'])
