"""Unique post-reboot owner; resumes unfinished epoch then original suite order."""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;PY=sys.executable
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def verify():
 for c in [Q/'CONTRACT.json',Q/'engineering_device_snapshot_20261101_v1/CONTRACT.json',E/'CONTRACT.json']:
  assert all(sha(Path(p))==h for p,h in js(c)['files'].items())
def main():
 lock=Path('/data/mlr_ifa/'+Q.name+'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'STARTED.json').exists();verify();assert js(E/'VALIDATION.json')['status']=='PASS_GPU_RECOVERY_AND_PAIRED_BRANCH_REPLAY'
 put(E/'STARTED.json',dict(pid=os.getpid(),time=time.time()))
 for group in ['L005','L020']:
  if group=='L020':assert (Q/'L005_EVALUATION_COMPLETE.json').exists()
  for phase in ['train','evaluate']:
   for seed in [41,42,43]:
    case=group+'_'+str(seed);r=Q/case;name='train_S' if phase=='train' else 'evaluate';verify()
    assert not (r/(name+'.exit')).exists()
    argv=[PY,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(E/'train_wrapper.py'),case] if phase=='train' else [PY,'-B',str(r/'evaluate.py')]
    logfile=E/(case+'_resume_train.log') if phase=='train' and case=='L005_41' else r/(name+'.log')
    with logfile.open('x') as f:
     p=subprocess.Popen(argv,cwd=r,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
     status=dict(phase=phase,suite=group,seed=seed,pid=p.pid,time=time.time(),controller=os.getpid(),engineering_contract=sha(E/'CONTRACT.json'),hardware='RTX3090',completion_source=str(E/'controller.exit'))
     put(E/'STATUS.json',status);put(Q/'STATUS.json',status);rc=p.wait()
    (r/(name+'.exit')).write_text(str(rc));assert rc==0,(case,phase,rc)
  assert all(len(js(Q/(group+'_'+str(k))/'OUTCOMES.json'))==200 for k in [41,42,43])
  put(Q/(group+'_EVALUATION_COMPLETE.json'),dict(status='PASS_SUITE_EXECUTION_PENDING_FINAL_CPU',episodes=600,time=time.time()))
 assert not (Q/'controller.exit').exists() and not (Q/'engineering_device_snapshot_20261101_v1/controller.exit').exists()
 put(Q/'STATUS.json',dict(phase='complete',time=time.time(),completion_source=str(E/'controller.exit')));(E/'controller.exit').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  put(E/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));(E/'controller.exit').write_text('1');raise
