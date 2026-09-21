"""One fixed queue: tests -> freeze -> GPU gates -> full train -> native eval -> audit."""
import os
import subprocess
import sys
import time
import traceback
from .common import *

def eval_command():
    return [sys.executable,'-u','-m','sample_qp_lifelong_v1.evaluate','seed=46',
        'benchmark_name=LIBERO_OBJECT','policy=complete_policy','lifelong=basic',
        'folder='+str(ROOT/'data'),'pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        'bddl_folder='+str(ROOT/'libero/libero/bddl_files'),
        'init_states_folder='+str(ROOT/'libero/libero/init_files'),
        'eval.n_eval=20','eval.num_procs=20','+start_task=6','+end_task=10','+aug=true',
        '+epoch_dir='+str(STORAGE/'checkpoints'),'+test_log_file='+str(RUN/'primary_ep99.log'),
        '+eval_epochs=[99]','+experiment_root='+str(RUN/'eval_primary_ep99'),'training=false']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'QUEUE_STARTED.json').exists(),'No duplicate queue or automatic rerun'
    write(OUT/'QUEUE_STARTED.json',dict(pid=os.getpid(),time=time.time(),seed=46,
          command=[sys.executable,'-m','sample_qp_lifelong_v1.queue']))
    env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
    # All real-policy gates and formal training use this identical, recorded environment.
    env['OPENBLAS_NUM_THREADS']='2'
    jobs=[('cpu_tests',[sys.executable,'-m','unittest','sample_qp_lifelong_v1.tests','-v'],1800),
          ('prepare',[sys.executable,'-u','-m','sample_qp_lifelong_v1.prepare'],3600),
          ('preflight',[sys.executable,'-u','-m','sample_qp_lifelong_v1.preflight'],7200),
          ('train',[sys.executable,'-u','-m','sample_qp_lifelong_v1.train'],120*3600+3600),
          ('evaluate',eval_command(),8*3600),
          ('summarize',[sys.executable,'-u','-m','sample_qp_lifelong_v1.summarize'],3600)]
    write(OUT/'QUEUE_PLAN.json',dict(jobs=[dict(stage=n,command=c,timeout=t) for n,c,t in jobs],
          environment={k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
    for name,cmd,timeout in jobs:
        write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',stage=name,time=time.time(),pid=os.getpid()))
        with (OUT/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',stage=name,time=time.time(),pid=os.getpid(),child_pid=child.pid))
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Frozen runtime budget exceeded: '+name)
        assert code==0,('Queue stopped; original logs retained',name,code)
        print('QUEUE_STAGE_COMPLETE',name,flush=True)
    write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'QUEUE_ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',time=time.time()))
        raise
