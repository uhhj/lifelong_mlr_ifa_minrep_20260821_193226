"""Ordered local experiment stages. No formal training entry point is invoked."""
import os,sys,subprocess,time
from .common import *
def publish(stage):
 files=[str(p.relative_to(ROOT)) for folder in [ROOT/'functional_component_counterfactual_v1',OUT] for p in folder.rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.png','.npz'] and p.name!='PUSH_STATUS.json' and '__pycache__' not in str(p)]
 subprocess.run(['git','add','--']+files,cwd=str(ROOT),check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=str(ROOT)).returncode:
  # Only these explicitly scoped paths are committed; no unrelated pre-existing staged changes.
  subprocess.run(['git','commit','-m','Record functional component counterfactual '+stage,'--']+files,cwd=str(ROOT),check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip()
 command=['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential','push','origin','codex/sample-qp-lifelong-v1']
 result=subprocess.run(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 write(OUT/'PUSH_STATUS.json',dict(stage=stage,head=head,returncode=result.returncode,output=result.stdout,remote='uhhj/lifelong_mlr_ifa_minrep_20260821_193226'))
 print('PUSH',stage,result.returncode,head,flush=True)
def main(wait_pid=None):
 if wait_pid:
  while True:
   try:os.kill(wait_pid,0)
   except ProcessLookupError:break
   time.sleep(10)
 if not (OUT/'SMOKE_COMPLETE.json').exists():
  status('QUEUE_STOPPED_SMOKE_NOT_PASS');return
 publish('smoke')
 for stage in ['offline','report','behavior','report']:
  if stage=='behavior' and read(OUT/'SUMMARY.json').get('MEASUREMENT_VALIDITY')!='PASS':
   status('QUEUE_STOPPED_MEASUREMENT_INVALID');return
  with (OUT/(stage+'_queue.log')).open('a') as f:
   result=subprocess.run([sys.executable,'-u','-m','functional_component_counterfactual_v1.run','--stage',stage,'--resume'],stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:
   publish(stage+'_failed');status('QUEUE_STOPPED',failed_stage=stage,returncode=result.returncode);return
  publish(stage)
 status('EXPERIMENT_COMPLETE')
if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else None)
