"""Resume only the frozen rollout budget and refresh reports; no offline rerun."""
import subprocess,sys
from .common import *
from .queue import publish

def main():
 assert len(list((OUT/'starts').glob('*/COMPLETE.json')))==36
 write(OUT/'BEHAVIORAL_REPORT.json',dict(status='RUNNING',trials_budget=480,initial_failed_launch_trials=0,formal_training_resumed=False))
 publish('behavior_import_fix')
 for stage in ['behavior','report']:
  with (OUT/(stage+'_resume.log')).open('a') as log:
   r=subprocess.run([sys.executable,'-u','-m','functional_component_counterfactual_v1.run','--stage',stage,'--resume'],stdout=log,stderr=subprocess.STDOUT)
  publish(stage+'_resume')
  if r.returncode:status('QUEUE_STOPPED',failed_stage=stage,returncode=r.returncode);return
 status('EXPERIMENT_COMPLETE')
if __name__=='__main__':main()
