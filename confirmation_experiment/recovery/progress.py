import json
import os
from confirmation_experiment.common import RUNS,read
from confirmation_experiment.recovery.common import WORK,AUDIT

def main():
    out={'original_failure_preserved':str(RUNS/'FAILED.json')}
    for n in ('STARTED.json','status.json','FAILED.json','COMPLETE.json'):
        if (WORK/n).exists():out[n]=read(WORK/n)
    if 'STARTED.json' in out:
        try:os.kill(out['STARTED.json']['pid'],0);out['recovery_pid_alive']=True
        except ProcessLookupError:out['recovery_pid_alive']=False
    out['training']={str(p.parent.relative_to(RUNS)):read(p) for p in sorted((RUNS/'formal').glob('seed*/*/TRAIN_PROGRESS.json'))}
    out['evaluation_completed_cells']={}
    for p in sorted((RUNS/'formal').glob('seed*/*')):
        state=p/'EVALUATION_COMPLETE.json';segment=WORK/f'primary_seed45_{p.name}_segment.log' if p.parent.name=='seed45' else p/'primary_ep99.log'
        if state.exists():n=10
        elif segment.exists():n=sum('[task ' in x for x in segment.read_text().splitlines())
        else:n=int((p/'TASK6_EVAL_READY.json').exists())
        out['evaluation_completed_cells'][str(p.relative_to(RUNS))]=n
    if (AUDIT/'SOFT45_BOUNDARY.json').exists():out['soft45_reconstruction']=read(AUDIT/'SOFT45_BOUNDARY.json')['status']
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
