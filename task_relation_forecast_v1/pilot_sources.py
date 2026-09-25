"""Prepare only the two prespecified source priors; target branches stay gated."""
import subprocess
import sys
import time
from .common import OUT, write


def main():
    for task in (6,7):
        for module in ('cache','source'):
            log=OUT/('PILOT_%s_45_%d.log'%(module,task))
            with log.open('a') as f:
                result=subprocess.run([sys.executable,'-u','-m','task_relation_forecast_v1.'+module,
                                       '--seed','45','--task',str(task)],stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                write(OUT/'STATUS.json',dict(protocol_status='RUNNING',stage='FAILED_STOP',module=module,
                      task=task,returncode=result.returncode,log=str(log),formal_training_resumed=False))
                raise RuntimeError('Pilot source process failed; see '+str(log))
    write(OUT/'STATUS.json',dict(protocol_status='RUNNING',stage='PILOT_SOURCES_READY',
          source_updates_completed=2000,target_updates_completed=0,rollouts_completed=0,
          diagnostic_training_performed=True,formal_training_resumed=False,
          remaining='Seal relation signals, shadow rollback and target pairing before target adaptation; then pilot evaluation and full matrix.'))


if __name__=='__main__':main()
