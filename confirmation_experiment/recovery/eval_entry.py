"""Retain completed task6 cells; execute the remaining original native calls."""
import os
import runpy
from pathlib import Path
import numpy as np
from confirmation_experiment.common import (ROOT,RUNS,read,write,rng_state,state_digest,atomic_torch_save,sha)
from confirmation_experiment.recovery.common import WORK,AUDIT,CELLS,install_abort_routing

def merge_segment(segment,canonical):
    from fixed_kappa_experiment.metrics import read_fixed_matrix,LINE
    read_fixed_matrix(segment)  # Reject incomplete/duplicate/out-of-protocol cells.
    previous=canonical.read_bytes() if canonical.exists() else b''
    old={}
    for line in previous.decode().splitlines():
        m=LINE.search(line)
        if m:
            key=(int(m[3]),int(m[1]));assert key not in old
            assert int(m[2])==99;old[key]=float(m[4])
    assert set(old).issubset({(6,6)}),'Unexpected previously completed cells; require a new audited recovery'
    added=[]
    for line in segment.read_text().splitlines():
        m=LINE.search(line)
        if not m:continue
        key=(int(m[3]),int(m[1]));value=float(m[4])
        if key in old:assert old[key]==value
        else:added.append(line+'\n')
    with canonical.open('a') as f:f.writelines(added)
    assert canonical.read_bytes().startswith(previous)
    read_fixed_matrix(canonical)
    return len(added)

def main():
    install_abort_routing()
    import libero.lifelong.metric as metric
    from confirmation_experiment.common import wait_for
    original=metric.evaluate_success
    run=Path(os.environ['CONFIRMATION_RUN_DIR']);reuse=os.environ.get('RECOVERY_REUSE_TASK6')=='1'
    if reuse:
        assert read(AUDIT/'EVAL_BOUNDARY_PARITY.json')['status']=='PASS'
        saved=read(run/'TASK6_EVAL_READY.json')
        assert saved['n_eval']==20 and saved['native_call_index']==1
        assert read(ROOT/'reports/fixed_kappa_confirmation_v2/TASK6_PARITY_SEED45.json')['status']=='PASS'
    records=[];native_count=0
    def observed(*args,**kw):
        nonlocal native_count
        index=len(records);stage,task=CELLS[index]
        assert kw['task_ids']==[task] and kw['cfg'].eval.n_eval==20
        if reuse and index==0:
            assert kw['cfg'].seed==45
            result=np.array(saved['success'])
            records.append({'cell':[stage,task],'source':'original completed 20-trial return','witness_sha256':sha(run/'TASK6_EVAL_READY.json')})
            return result
        before=rng_state();result=original(*args,**kw);native_count+=1
        after=rng_state()
        destination=run/'eval_recovery_states'/('cell_%02d.pt'%(index+1))
        atomic_torch_save(destination,{'cell':[stage,task],'seed':kw['cfg'].seed,'success':list(map(float,result)),
              'rng_before':before,'rng_after':after,'trials':20,'native_call_completed':True})
        records.append({'cell':[stage,task],'source':'native evaluation','state_path':str(destination.relative_to(ROOT)),
                        'state_sha256':sha(destination)})
        if index==0:
            write(run/'TASK6_EVAL_READY.json',{'success':list(map(float,result)),
                  'n_eval':20,'rng_sha256':state_digest(after),'native_call_index':1,
                  'note':'This same call resumes and is logged once by original test.py.'})
            wait_for(run/'RELEASE_EVAL',WORK/'FAILED.json')
        return result
    metric.evaluate_success=observed
    runpy.run_path(str(ROOT/'libero/lifelong/test.py'),run_name='__main__')
    assert len(records)==10 and native_count==(9 if reuse else 10)
    added=None
    if reuse:
        arm=run.name;segment=WORK/f'primary_seed45_{arm}_segment.log'
        added=merge_segment(segment,run/'primary_ep99.log')
    write(run/'EVALUATION_COMPLETE.json',{'native_evaluation_calls':native_count+int(reuse),
          'current_process_native_calls':native_count,'reused_previously_completed_calls':int(reuse),
          'trials_per_cell':20,'unique_completed_cells':10,'added_canonical_rows':added,'records':records,
          'recovery_note':'Previously interrupted incomplete cells are replayed under the original protocol; completed task6 trials are reused.'})

if __name__=='__main__':main()
