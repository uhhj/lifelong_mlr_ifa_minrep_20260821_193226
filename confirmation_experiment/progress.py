"""Read-only queue/child progress; does not wake or restart any process."""
import json
import os
from confirmation_experiment.common import RUNS,REPORTS,read
def main():
    data={}
    for k,p in [('preflight',REPORTS/'PREFLIGHT_STATUS.json'),('status',RUNS/'status.json'),
                ('started',RUNS/'STARTED.json'),('complete',RUNS/'COMPLETE.json'),('failed',RUNS/'FAILED.json')]:
        if p.exists():data[k]=read(p)
    if 'started' in data:
        try:os.kill(data['started']['pid'],0);data['queue_pid_alive']=True
        except ProcessLookupError:data['queue_pid_alive']=False
    data['training_progress']={}
    for p in sorted((RUNS/'formal').glob('seed*/*/TRAIN_PROGRESS.json')):
        data['training_progress'][str(p.parent.relative_to(RUNS))]=read(p)
    print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
