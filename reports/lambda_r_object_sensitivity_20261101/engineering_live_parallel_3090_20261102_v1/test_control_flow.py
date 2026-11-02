"""Exercise actual live-adoption main with fake processes; no signals or experiments."""
import ast,json,tempfile,signal,types,hashlib
from pathlib import Path
signal=types.SimpleNamespace(SIGSTOP=19,SIGKILL=9)
E0=Path(__file__).resolve().parent
source=(E0/'controller.py').read_text(encoding='utf8');tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
# Redirect the only absolute root-lock path to this test's temporary directory.
for n in ast.walk(main):
 if isinstance(n,ast.Constant) and n.value=='/data/mlr_ifa/':n.value=''
code=compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'actual_controller_main','exec')
results=[]
for first,rc42,rc43 in [('42',0,0),('43',0,0),('42',1,0),('43',0,1)]:
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
  Q=Path(td);E=Q/'eng';E.mkdir();(Q/'L005_42').mkdir();(E/'WATCHDOG_READY.json').write_text('{}')
  ids={'root':{'pid':4347,'start':'1'},'parallel':{'pid':8608,'start':'2'},'watchdog':{'pid':8609,'start':'3'}};g={'controllers':ids,'child':{'pid':4689,'start':'4'},'training':{'pid':4694,'start':'5'}}
  states={4347:'T',8608:'S',8609:'S',4689:'R',4694:'R'};ticks=[0];events=[];live={};ends={'42':2 if first=='42' else 4,'43':2 if first=='43' else 4}
  def js(p):
   name=Path(p).name
   if name=='GUARD.json':return g
   if name=='PARALLEL_ADMISSION.json':return {'status':'PASS_EIGHT_STAGE_PARALLEL_EXACT_AND_TIMING'}
   if name=='STARTUP_LOCK_VALIDATION.json':return {'status':'PASS_ZERO_UPDATE_EXACT_STARTUP_LOCK'}
   if name=='TRAIN_STATUS.json':return {'phase':'train','pid':4694,'epoch_complete':45}
   if name=='OUTCOMES.json':return [None]*200
   raise AssertionError(str(p))
  def same(i):
   pid=i['pid'];state=states[pid]
   if pid==4689 and ticks[0]>=ends['42']:state='Z'
   return dict(i,state=state,exit_code=rc42<<8)
  def kill(pid,sig):
   assert pid in [4347,8608,8609];events.append(('signal',pid,int(sig)));states[pid]='T' if sig==signal.SIGSTOP else 'Z'
  def launch(case,phase):events.append(('launch',case,phase));assert case not in live;live[case]=(types.SimpleNamespace(pid=9001),phase)
  def finish(case):
   if ticks[0]<ends['43']:return None
   del live[case];events.append(('finish',case,rc43));return rc43
  def one(case,phase):events.append(('one',case,phase));assert not live and scope['adopted'] is None
  def pool(cases):events.append(('pool',tuple(cases)))
  def put(p,x):Path(p).write_text(json.dumps(x))
  class LockPath:
   def __init__(self,p):self.p=Q/'test_root.lock'
   def open(self,*args):return self.p.open(*args)
  scope=dict(E=E,Q=Q,PY='fake',live=live,adopted=None,fcntl=types.SimpleNamespace(flock=lambda *x:None,LOCK_EX=1,LOCK_NB=2),verify=lambda:None,js=js,same=same,prefix_guard=lambda g:None,put=put,proc=lambda p:{'pid':p},os=types.SimpleNamespace(getpid=lambda:999,kill=kill),signal=signal,subprocess=types.SimpleNamespace(Popen=lambda *a,**k:types.SimpleNamespace(pid=888),DEVNULL=None,STDOUT=None),time=types.SimpleNamespace(time=lambda:ticks[0],sleep=lambda s:ticks.__setitem__(0,ticks[0]+1)),launch=launch,finish=finish,state=lambda p:None,complete=lambda case:events.append(('complete',case)),Path=LockPath,one=one,train_pool=pool)
  exec(code,scope)
  passed=True
  try:scope['main']()
  except AssertionError:
   assert rc42 or rc43;passed=False
  assert sum(x[:2]==('launch','L005_43') for x in events)==1
  assert not any(x[0]=='signal' and x[1] in [4689,4694] for x in events)
  assert (E/'HANDOFF_COMPLETE.json').exists() and (Q/'L005_42/train_S.exit').read_text()==str(rc42)
  assert not (Q/'controller.exit').exists()
  if rc42 or rc43:assert not any(x[0] in ['one','pool'] for x in events)
  else:
   assert (E/'controller.exit').read_text()=='0'
   assert [x[0] for x in events if x[0] in ['one','pool']]==['one']*3+['pool']+['one']*3
  results.append(dict(first_finisher=first,rc42=rc42,rc43=rc43,success=passed,events=events))
# Guard handler must terminate even if stderr is unavailable.
body=next(n.value for n in ast.walk(ast.parse((E0/'dispatch_live.py').read_text(encoding='utf8'))) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'finally:os._exit(86)' in n.value)
assert 'finally:os._exit(86)' in body
result=dict(status='PASS_ACTUAL_MAIN_SIMULATED_ORDER_AND_FAILURES',cases=results,production_signals=0,optimizer_calls=0,environments=0,controller_sha256=hashlib.sha256(source.encode()).hexdigest())
(E0/'LOCAL_CONTROL_FLOW_REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(result['status'])
