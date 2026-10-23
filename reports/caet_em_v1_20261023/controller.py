import os,sys,subprocess,time,fcntl,traceback
from pathlib import Path
from caet_em_exp.common import *
def phase(name,argv):
 verify();capacity();out=ROOT/'phases';out.mkdir(exist_ok=True)
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.Popen([sys.executable,'-m','caet_em_exp.'+argv[0],*argv[1:]],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+':'+str(SOURCE),TOKENIZERS_PARALLELISM='false',MUJOCO_GL='egl'))
  write(ROOT/'RUN_STATUS.json',dict(status='PARTIAL',phase=name,pid=p.pid,controller=os.getpid(),updated_unix=time.time()));rc=p.wait()
 (out/(name+'.exit')).write_text(str(rc)+'\n');assert rc==0,(name,rc)
def main():
 lock=open('/data/mlr_ifa/physical_effect_probe.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (ROOT/'EVENTS.jsonl').exists(),'No blind controller restart'
 try:
  phase('prepare',['prepare'])
  if not (ROOT/'evidence/task6/TEMPORAL_BINDING_AUDIT.json').exists():phase('evidence6',['evidence','6'])
  else:verify();event('evidence_reused',task=6,labels_sha256=sha(ROOT/'evidence/task6/labels.pth'))
  phase('codec_admission',['admission','codec']);phase('codec',['train_codec'])
  for arm in ARMS:
   phase(arm+'_admission',['admission',arm])
   for task in range(6,10):
    if not (ROOT/'evidence'/('task%d'%task)/'TEMPORAL_BINDING_AUDIT.json').exists():phase('evidence%d'%task,['evidence',str(task)])
    phase('%s_train%d'%(arm,task),['train',arm,str(task)])
    phase('%s_diagnose%d'%(arm,task),['diagnose',arm,str(task)])
   phase(arm+'_evaluate',['evaluate',arm]);phase(arm+'_audit',['audit',arm])
  write(ROOT/'RUN_STATUS.json',dict(status='SERVER_COMPLETE_LOCAL_ACCEPTANCE_PENDING',controller=os.getpid(),updated_unix=time.time()));(ROOT/'controller.exit').write_text('0\n')
 except BaseException as e:
  write(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));(ROOT/'controller.exit').write_text('1\n');raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
