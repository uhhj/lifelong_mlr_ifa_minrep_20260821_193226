"""Finite authorized suite sequence; never restarts a run or selects outcomes."""
import os,json,hashlib,time,subprocess,fcntl,traceback
from pathlib import Path
Q=Path(__file__).resolve().parent;PY='/data/mlr_ifa/env/bin/python'
SUITES=['OBJECT','GOAL']
def root(s):return Q.parent/('S_Q1_'+s+'41_42_43_20261109_rerun_v1')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
 p=Q/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def verify():
 c=js(Q/'CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
def events(kind,**kw):
 with (Q/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kw))+'\n')
def main():
 assert not (Q/'controller.exit').exists() and not (Q/'EVENTS.jsonl').exists()
 lock=(Q/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for suite in SUITES:
   verify();r=root(suite);assert not (r/'controller.exit').exists() and not (r/'controller.log').exists()
   with (r/'controller.log').open('x') as f:
    p=subprocess.Popen([PY,'-B',str(r/'controller_parallel.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    write('STATUS.json',dict(status='RUNNING',suite=suite,suite_controller=p.pid,controller=os.getpid(),observed=time.time()))
    events('suite_launch',suite=suite,pid=p.pid);code=p.wait()
   events('suite_exit',suite=suite,pid=p.pid,code=code);assert code==0 and (r/'controller.exit').read_text().strip()=='0'
   assert js(r/'Q1_EVALUATION_COMPLETE.json')['episodes']==600
   for seed in [41,42,43]:
    c=r/('Q1_'+str(seed));assert (c/'train_S.exit').read_text().strip()=='0' and (c/'evaluate.exit').read_text().strip()=='0'
    assert len(js(c/'EPOCHS_S.json'))==400 and len(js(c/'OUTCOMES.json'))==200
   for proc in Path('/proc').glob('[0-9]*'):
    try:assert str(r) not in (proc/'cmdline').read_bytes().decode(errors='replace'),('Residual suite process',proc.name)
    except OSError:pass
   write(suite+'_NATURAL_COMPLETE.json',dict(observed=time.time(),episodes=600,training_epochs=1200,suite_controller_exit=0,independent_cpu_acceptance='PENDING'))
  verify();write('STATUS.json',dict(status='NATURAL_COMPLETE_PENDING_INDEPENDENT_CPU_ACCEPTANCE',observed=time.time(),episodes=1200))
  with (Q/'controller.exit').open('x') as f:f.write('0')
 except BaseException:
  write('FAILURE.json',dict(observed=time.time(),error=traceback.format_exc()))
  with (Q/'controller.exit').open('x') as f:f.write('1')
  raise
 finally:lock.close()
if __name__=='__main__':main()
