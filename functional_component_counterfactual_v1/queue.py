"""Ordered local experiment stages. No formal training entry point is invoked."""
import os,sys,subprocess,time
from .common import *
def main(wait_pid=None):
 if wait_pid:
  while True:
   try:os.kill(wait_pid,0)
   except ProcessLookupError:break
   time.sleep(10)
 if not (OUT/'SMOKE_COMPLETE.json').exists():
  status('QUEUE_STOPPED_SMOKE_NOT_PASS');return
 for stage in ['offline','report','behavior','report']:
  if stage=='behavior' and read(OUT/'SUMMARY.json').get('MEASUREMENT_VALIDITY')!='PASS':
   status('QUEUE_STOPPED_MEASUREMENT_INVALID');return
  with (OUT/(stage+'_queue.log')).open('a') as f:
   result=subprocess.run([sys.executable,'-u','-m','functional_component_counterfactual_v1.run','--stage',stage,'--resume'],stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:status('QUEUE_STOPPED',failed_stage=stage,returncode=result.returncode);return
 status('EXPERIMENT_COMPLETE')
if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else None)
