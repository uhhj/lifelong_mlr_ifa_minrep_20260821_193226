"""One fixed continuation -> ep99 lifecycle evaluation -> summary; no restart."""
import fcntl, os, signal, subprocess, sys, time, traceback
from .common import *

def commands():
    package='sample_qp_lifelong_v1_continuation_v4'
    evaluation=[sys.executable,'-u','-m',package+'.evaluate',
        'seed=46','benchmark_name=LIBERO_OBJECT','policy=complete_policy','lifelong=basic',
        'folder='+str(ROOT/'data'),'pretrain='+str(ROOT/'checkpoints/object_multitask_model_ep99.pth'),
        'bddl_folder='+str(ROOT/'libero/libero/bddl_files'),
        'init_states_folder='+str(ROOT/'libero/libero/init_files'),
        'eval.n_eval=20','eval.num_procs=20','+start_task=6','+end_task=10','+aug=true',
        '+epoch_dir='+str(STORAGE/'checkpoints'),'+test_log_file='+str(RUN/'primary_ep99.log'),
        '+eval_epochs=[99]','+experiment_root='+str(RUN/'eval_primary_ep99'),'training=false']
    return [('train',[sys.executable,'-u','-m',package+'.train','--resume'],121*3600),
            ('evaluate',evaluation,8*3600),
            ('summarize',[sys.executable,'-u','-m',package+'.summarize'],3600)]

def main():
    assert read(OUT/'EPOCH_RECOVERY_MIGRATION.json')['status']=='PASS'
    verify()
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'QUEUE_STARTED.json').exists(),'Queue already launched; no automatic retry'
    env=os.environ.copy()
    for secret in ('GH_TOKEN','GITHUB_TOKEN','PUSH_TOKEN'):env.pop(secret,None)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               WANDB_MODE='disabled',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless')
    plan=commands()
    write(OUT/'QUEUE_STARTED.json',dict(pid=os.getpid(),time=time.time(),commands=plan,
        implementation_sha=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        execution_sha256=sha(OUT/'EXECUTION_FROZEN.json'),formal_training_resumed=True))
    for name,cmd,timeout in plan:
        if name=='evaluate':
            assert read(OUT/'V4_NUMERICAL_RECOVERY_PASS.json')['status']=='V4_NUMERICAL_RECOVERY_PASS'
        log=OUT/(name+'_formal.log')
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',phase=name,pid=os.getpid(),
                child_pid=proc.pid,log=str(log),time=time.time(),formal_training_resumed=True))
            try:code=proc.wait(timeout=timeout)
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        assert code==0,('%s failed; no retry, see %s'%(name,log),code)
        print('PHASE_COMPLETE',name,flush=True)
    write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',time=time.time(),formal_training_resumed=True))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'QUEUE_FAILURE.json',dict(status='FAILED_STOP',time=time.time(),error=traceback.format_exc(),
            formal_training_resumed=(OUT/'QUEUE_STARTED.json').exists()))
        raise
