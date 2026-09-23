"""Run the frozen diagnostic; commit/push only its small outputs after exit."""
import os,subprocess,sys,time,fcntl
from .common import *
def main():
 lock=(OUT/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 begin=time.time();rc=1
 expected=read(OUT/'EXECUTION_FREEZE.json')
 for rel,value in expected['files'].items():assert sha(ROOT/rel)==value,rel
 env=os.environ.copy();token=env.pop('GH_TOKEN',None)
 # The measurement child does not receive Git credentials.
 with (OUT/'run.log').open('a') as log:
  p=subprocess.Popen([sys.executable,'-u','-m','qp_corrected_update_fidelity_v1.run','--phase','all'],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
  write(OUT/'QUEUE_STARTED.json',dict(queue_pid=os.getpid(),worker_pid=p.pid,started_unix=begin,formal_training_resumed=False,implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip()))
  rc=p.wait()
 write(OUT/'QUEUE_COMPLETE.json',dict(returncode=rc,seconds=time.time()-begin,formal_training_resumed=False,scientific_complete=(rc==0)))
 # User requested a push after every run. No historical paths or large states staged.
 allowed=['qp_corrected_update_fidelity_v1','reports/qp_corrected_update_fidelity_v1']
 for name in allowed:subprocess.run(['git','add','--',name],cwd=str(ROOT),check=True)
 changed=subprocess.run(['git','diff','--cached','--quiet','--']+allowed,cwd=str(ROOT)).returncode!=0
 if changed:subprocess.run(['git','commit','-m','Report QP corrected-update fidelity diagnostic run','--']+allowed,cwd=str(ROOT),check=True)
 pushenv=os.environ.copy()
 if token:pushenv['GH_TOKEN']=token
 pushenv['GIT_TERMINAL_PROMPT']='0'
 push=subprocess.run(['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential','push','origin','codex/sample-qp-lifelong-v1'],cwd=str(ROOT),env=pushenv,capture_output=True,text=True)
 write(OUT/'PUSH_STATUS.json',dict(returncode=push.returncode,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),output=push.stdout+push.stderr))
 sys.exit(rc if rc else push.returncode)
if __name__=='__main__':main()
