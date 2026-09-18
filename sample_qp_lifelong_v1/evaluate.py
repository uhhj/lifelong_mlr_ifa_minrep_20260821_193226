"""Passive trial recorder around unchanged native test.py; no RNG reseeding."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('EGL_PLATFORM','surfaceless')
import hashlib
import runpy
import time
from .common import *

class Observer:
    def __init__(self,inner):
        self.inner=inner;self.rounds=[];self.active=None
    def __getattr__(self,name):return getattr(self.inner,name)
    def finish(self):
        if self.active is not None:
            self.active['actions_sha256']=self.ah.hexdigest();self.active['done_sha256']=self.dh.hexdigest()
            self.rounds.append(self.active);self.active=None
    def set_init_state(self,states):
        from cb_ifa_experiment.gpu_runtime import digest
        self.finish();self.ah=hashlib.sha256();self.dh=hashlib.sha256()
        self.active=dict(round=len(self.rounds),init_state_sha256=digest(states),step=-5,
                         successes=[False]*10,first_success_step=[None]*10)
        return self.inner.set_init_state(states)
    def step(self,actions):
        import numpy as np
        saved=np.asarray(actions).copy();result=self.inner.step(actions)
        done=np.asarray(result[2],dtype=bool);a=self.active;a['step']+=1
        self.ah.update(saved.tobytes());self.dh.update(done.tobytes())
        if a['step']>0:
            for i,v in enumerate(done):
                if v and not a['successes'][i]:a['first_success_step'][i]=a['step']
                a['successes'][i]|=bool(v)
        return result
    def close(self):self.finish();return self.inner.close()

def main():
    import numpy as np
    from cb_ifa_experiment.gpu_runtime import rng_state,restore_rng,digest,atomic_save,torch
    from libero.lifelong import metric
    verify();assert read(RUN/'TRAINING_COMPLETE.json')['status']=='PASS'
    dest=RUN/'evaluation';dest.mkdir(exist_ok=True)
    native=metric.evaluate_success;factory=metric.SubprocVectorEnv;instances=[];calls=[]
    cells=[(s,t) for s in TASKS for t in range(6,s+1)]
    def make(*a,**kw):
        wrapped=Observer(factory(*a,**kw));instances.append(wrapped);return wrapped
    def observed(*a,**kw):
        stage,task=cells[len(calls)];cfg=kw['cfg'];algo=kw['algo']
        assert kw['task_ids']==[task] and cfg.eval.n_eval==20 and cfg.seed==SEED
        assert cfg.policy.use_language and not cfg.policy.use_state
        assert not cfg.policy.policy_head.network_kwargs.low_eval_noise
        checkpoint=STORAGE/'checkpoints'/('mlrifa_%d_model_ep99.pth'%stage)
        saved=torch.load(checkpoint,map_location='cpu')['state_dict']
        expected={k[7:]:v for k,v in saved.items()}
        actual=algo.policy.state_dict()
        assert expected.keys() <= actual.keys()
        assert digest(expected)==digest({k:actual[k].cpu() for k in expected})
        before=digest(rng_state());full_before=digest(actual)
        cell=dest/('stage%d_task%d.json'%(stage,task));rngfile=cell.with_suffix('.rng.pt')
        if cell.exists():
            old=read(cell);assert old['rng_before']==before and old['checkpoint_sha256']==sha(checkpoint)
            assert sha(rngfile)==old['rng_sha256'];restore_rng(torch.load(rngfile,map_location='cpu'))
            result=np.array([old['success_rate']]);calls.append(old);return result
        instances.clear();started=time.time();result=native(*a,**kw)
        rounds=[r for item in instances for r in item.rounds]
        assert len(instances)==1 and len(rounds)==2
        trials=[dict(trial=r['round']*10+i,success=int(v),first_success_step=r['first_success_step'][i])
                for r in rounds for i,v in enumerate(r['successes'])]
        assert len(trials)==20 and sum(r['success'] for r in trials)/20==float(result[0])
        assert digest(algo.policy.state_dict())==full_before
        after=rng_state();atomic_save(rngfile,after)
        row=dict(stage=stage,task=task,trials=trials,rounds=rounds,success_rate=float(result[0]),
                 rng_before=before,rng_after=digest(after),rng_sha256=sha(rngfile),
                 checkpoint_sha256=sha(checkpoint),seconds=time.time()-started)
        write(cell,row);calls.append(row)
        write(OUT/'STATUS.json',dict(status='EVALUATING',completed_cells=len(calls),total_cells=10,
                                   stage=stage,task=task,time=time.time()))
        return result
    metric.SubprocVectorEnv=make;metric.evaluate_success=observed
    try:runpy.run_path(str(ROOT/'libero/lifelong/test.py'),run_name='__main__')
    finally:metric.SubprocVectorEnv=factory;metric.evaluate_success=native
    assert len(calls)==10
    write(RUN/'EVALUATION_COMPLETE.json',dict(status='PASS',cells=10,trials=200,
          success_matrix={str((r['stage'],r['task'])):r['success_rate'] for r in calls}))

if __name__=='__main__':main()
