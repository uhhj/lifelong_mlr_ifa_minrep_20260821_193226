"""One durable preparation-and-execution queue for the independent4090 cohort."""
import os
import shutil
import subprocess
import sys
import time
import traceback
from gradient_relation_experiment.hardware_transition.entry import ROOT,OLD,OUT,COHORT,read,write,sha,original_integrity


def run():
    OUT.mkdir(exist_ok=True);assert not (OUT/'QUEUE_STARTED.json').exists()
    original_integrity();assert read(OUT/'TRANSITION_PROBE.json')['status']=='CROSS_HARDWARE_CONTINUATION_NOT_EQUIVALENT'
    write(OUT/'QUEUE_STARTED.json',{'pid':os.getpid(),'time':time.time(),'seeds':[45,46,47],'points':120,
        'cohort':COHORT,'old_3090_points_preserved':24,'phase':'preflight'})
    for name in ['DESIGN.json.gz','PREPARED.json','PROTOCOL.md']:
        target=OUT/name;assert not target.exists();shutil.copyfile(OLD/name,target)
    shutil.copyfile(Path(__file__).with_name('HARDWARE_PROTOCOL.md'),OUT/'HARDWARE_PROTOCOL.md')
    jobs=[('preflight',None),('control',None),('analysis_test',None),('run',45),('run',46),('run',47),('summary',None)]
    try:
        for action,seed in jobs:
            if action=='run' and seed==45:
                for name in ['PREFLIGHT_COMPLETE.json','NATIVE_CONTROL_COMPLETE.json','ANALYSIS_TEST.json']:assert read(OUT/name)['status']=='PASS'
                old=read(OLD/'FROZEN_EXECUTION.json');files=dict(old['files'])
                for f in Path(__file__).parent.iterdir():
                    if f.is_file() and f.suffix in ('.py','.md'):files[str(f.relative_to(ROOT))]=sha(f)
                for f in [OUT/'DESIGN.json.gz',OUT/'HARDWARE_PROTOCOL.md',OUT/'TRANSITION_PROBE.json',OUT/'recovery_probe_seed45/WITNESS_COMPARISON.json']:
                    files[str(f.relative_to(ROOT))]=sha(f)
                write(OUT/'FROZEN_EXECUTION.json',{'carrier_source':COHORT,'files':files,'frozen_at':time.time(),
                    'original_implementation_sha':old['implementation_sha'],'predefined_design_sha256':sha(OUT/'DESIGN.json.gz'),
                    'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'old_3090_points_not_pooled':24,'seeds':[45,46,47],'target_points':120})
            name=action+(f'_seed{seed}' if seed is not None else '')
            cmd=[sys.executable,'-m','gradient_relation_experiment.hardware_transition.entry',action]
            if seed is not None:cmd+=['--seed',str(seed)]
            with (OUT/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(OUT/'QUEUE_STATUS.json',{'status':'RUNNING','phase':action,'seed':seed,'child_pid':child.pid,'queue_pid':os.getpid(),'time':time.time()})
                assert child.wait()==0,f'{name} failed; outputs retained, no automatic retry'
        for rel,h in read(OUT/'PRESERVATION.json')['source_files'].items():assert sha(ROOT/rel)==h
        write(OUT/'QUEUE_STATUS.json',{'status':'COMPLETE','time':time.time(),'points_completed':120,'old_3090_preservation':'PASS'})
    except Exception:
        write(OUT/'QUEUE_FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()})
        write(OUT/'QUEUE_STATUS.json',{'status':'FAILED_INTEGRITY','time':time.time()});raise


from pathlib import Path
if __name__=='__main__':run()
