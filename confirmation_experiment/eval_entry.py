"""Pause native test.py after its first actual 20-trial cell; never repeat it."""
import os
from pathlib import Path

def main():
    import libero.lifelong.metric as metric
    from confirmation_experiment.common import write,wait_for,RUNS,state_digest,rng_state
    original=metric.evaluate_success
    count=0
    run=Path(os.environ['CONFIRMATION_RUN_DIR'])
    def observed_success(*args,**kwargs):
        nonlocal count
        result=original(*args,**kwargs)
        count+=1
        if count==1:
            assert list(kwargs['task_ids'])==[6]
            write(run/'TASK6_EVAL_READY.json',{'success':list(map(float,result)),
                  'n_eval':int(kwargs['cfg'].eval.n_eval),'rng_sha256':state_digest(rng_state()),
                  'native_call_index':1,'note':'This same call resumes and is logged once by original test.py.'})
            wait_for(run/'RELEASE_EVAL',RUNS/'FAILED.json')
        return result
    metric.evaluate_success=observed_success
    import runpy
    from confirmation_experiment.common import ROOT
    runpy.run_path(str(ROOT/'libero/lifelong/test.py'),run_name='__main__')
    assert count==10,'Original evaluation matrix must contain ten cells'
    write(run/'EVALUATION_COMPLETE.json',{'native_evaluation_calls':count,'trials_per_cell':20})
if __name__=='__main__':main()
