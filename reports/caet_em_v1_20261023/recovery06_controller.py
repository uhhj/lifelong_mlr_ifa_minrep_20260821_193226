"""Versioned orchestration only; use frozen engineering05 native --resume."""
import os,sys,subprocess,time,fcntl,traceback
from pathlib import Path
Q=Path(os.environ['EM_RECOVERY_ROOT']);ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
def validate():
 assert verify()==js(Q/'RECOVERY_MANIFEST.json')['science_contract']
 for path,h in js(Q/'EXECUTION_CONTRACT.json')['files'].items():assert sha(path)==h,path
 capacity()
def phase(name,cmd):
 validate();out=ROOT/'phases';log=out/(name+'.log')
 if log.exists():log=out/(name+'_recovery06.log')
 assert not log.exists() and not (out/(name+'.exit')).exists(),name
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
  write(ROOT/'RUN_STATUS.json',dict(status='PARTIAL',phase=name,pid=p.pid,controller=os.getpid(),updated_unix=time.time(),recovery_root=str(Q)))
  rc=p.wait()
 (out/(name+'.exit')).write_text(str(rc)+'\n');assert rc==0,(name,rc)
def module(name,*args):return [sys.executable,'-B','-m','caet_em_exp.'+name,*args]
lock=open('/data/mlr_ifa/physical_effect_probe.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 assert not (Q/'STARTED.json').exists(),'No blind recovery-controller restart'
 validate();write(Q/'STARTED.json',dict(unix=time.time(),pid=os.getpid()))
 assert not (ROOT/'controller.exit').exists()
 phase('B2_admission_recovery06',module('admission','B2'))
 phase('B2_exact_recovery06',[sys.executable,'-B',str(Q/'recovery06_admission.py')])
 assert js(Q/'RECOVERY_ADMISSION.json')['status']=='PASS'
 write(ROOT/'RECOVERY06_ADMISSION.json',js(Q/'RECOVERY_ADMISSION.json'))
 phase('B2_train8',module('train','B2','8','--resume'))
 history=js(arm_root('B2')/'epochs/task8.json');old=js(Q/'provenance/task8_epochs.json')
 assert history[:90]==old and len(history)==100
 write(ROOT/'RECOVERY06_RESOLUTION.json',dict(status='PASS_COMPLETE_EPOCH_CONTINUATION',preserved_prefix_epochs=90,preserved_prefix_steps=66210,task8_final_steps=69300,repeated_completed_epochs=0,recovery_execution_contract=sha(Q/'EXECUTION_CONTRACT.json'),admission_sha256=sha(Q/'RECOVERY_ADMISSION.json'),recovery_root=str(Q)))
 phase('B2_diagnose8',module('diagnose','B2','8'))
 phase('B2_train9',module('train','B2','9'))
 phase('B2_diagnose9',module('diagnose','B2','9'))
 phase('B2_evaluate',module('evaluate','B2'))
 phase('B2_audit',module('audit','B2'))
 write(ROOT/'RUN_STATUS.json',dict(status='SERVER_COMPLETE_LOCAL_ACCEPTANCE_PENDING',controller=os.getpid(),updated_unix=time.time(),recovery_root=str(Q)))
 (ROOT/'controller.exit').write_text('0\n');(Q/'controller.exit').write_text('0\n')
except BaseException as e:
 write(Q/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()))
 (Q/'controller.exit').write_text('1\n');(ROOT/'controller.exit').write_text('1\n');raise
finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
