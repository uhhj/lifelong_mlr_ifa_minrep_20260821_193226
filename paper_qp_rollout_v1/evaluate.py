import os,time,sys,hashlib,traceback
from .common import *
from cb_ifa_experiment.gpu_runtime import torch,np,seed_all,rng_state,digest,eval_config
class Observer:
    """Paired round RNG boundary; optional passive array retention."""
    def __init__(self,inner,cell,out,trace):
        self.inner=inner;self.cell=cell;self.out=out;self.trace=trace;self.rounds=[];self.active=None
    def __getattr__(self,n):return getattr(self.inner,n)
    def set_init_state(self,states):
        self.flush();r=len(self.rounds)
        seed_all(self.cell['round_policy_seeds'][r]);self.arrays=[];self.flags=[];self.stepnums=[]
        self.ah=hashlib.sha256();self.dh=hashlib.sha256()
        self.active={'round':r,'init_sha256':digest(states),'policy_seed':self.cell['round_policy_seeds'][r],'policy_rng_before':digest(rng_state()),'count':-5,'success':[False]*10,'first':[None]*10}
        obs=self.inner.set_init_state(states)
        # Observation fingerprint catches renderer/state differences without storing images.
        self.active['initial_obs_sha256']=observation_digest(obs)
        return obs
    def step(self,actions):
        saved=np.asarray(actions).copy();assert saved.shape==(10,7) and np.isfinite(saved).all()
        result=self.inner.step(actions);done=np.asarray(result[2],dtype=bool).copy();a=self.active;a['count']+=1
        self.ah.update(saved.tobytes());self.dh.update(done.tobytes())
        if self.trace:self.arrays.append(saved);self.flags.append(done);self.stepnums.append(a['count'])
        if a['count']>0:
            for lane,v in enumerate(done):
                if v and not a['success'][lane]:a['first'][lane]=a['count']
                a['success'][lane]|=bool(v)
        return result
    def flush(self):
        if self.active is None:return
        a=self.active;a.update(action_sha256=self.ah.hexdigest(),done_sha256=self.dh.hexdigest(),policy_rng_after=digest(rng_state()))
        if self.trace:
            path=self.out/('round'+str(a['round'])+'.npz')
            np.savez_compressed(path,actions=np.stack(self.arrays),done=np.stack(self.flags),steps=np.asarray(self.stepnums))
            a['trace_file_sha256']=sha(path)
        self.rounds.append(a);self.active=None
    def close(self):self.flush();return self.inner.close()
def run(index,tag=None,trace=True):
    verify();p=read(OUT/'PROTOCOL_FROZEN.json');cell=p['cells'][index]
    out=OUT/'evaluation'/cell['point']/('task'+str(cell['task']))/cell['arm'] if tag is None else OUT/'engineering'/tag
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        c=read(out/'COMPLETE.json')
        for f,h in c['files'].items():assert sha(out/f)==h
        print('PRESERVED_COMPLETED_CELL',index,flush=True);return
    assert not (out/'STARTED.json').exists(),'Interrupted cell requires explicit faithful-resume audit; no silent rerun'
    write(out/'STARTED.json',{'cell':cell,'time':time.time(),'pid':os.getpid(),'engineering':tag})
    start=time.time();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=True
    assert torch.cuda.get_device_name(0)==p['resources']['GPU']
    import robomimic.utils.obs_utils as ObsUtils
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import get_task_embs
    from libero.lifelong import metric
    seed=cell['environment_seed'];cfg=eval_config(seed,out);frozen=read(OUT/f'EVAL_CONFIG_seed{seed}.json')
    for k in ['policy','data','eval']:assert dict(cfg[k])==frozen[k],k
    write(out/'CONFIG.json',cfg)
    seed_all(seed);ObsUtils.initialize_obs_utils_with_obs_specs({'obs':dict(cfg.data.obs.modality)})
    bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    bench.set_task_embs(get_task_embs(cfg,[bench.get_task(i).language for i in range(bench.n_tasks)]))
    algo=get_algo_class(cfg.lifelong.algo)(bench.n_tasks,cfg).to('cuda:0')
    rec=read(OUT/'recovery'/cell['point']/'COMPLETE.json')['checkpoints'][cell['arm']]
    assert sha(rec['path'])==rec['sha256'];ck=torch.load(rec['path'],map_location='cpu')
    assert digest(ck['state_dict'])==rec['model_hash']
    state={k[7:] if k.startswith('module.') else k:v for k,v in ck['state_dict'].items()}
    expected={k for k in algo.policy.state_dict() if k.startswith(('temporal_transformer.','policy_head.'))}
    assert set(state)==expected,('endpoint keys',set(state)^expected)
    frozen_before=digest({k:v.cpu() for k,v in algo.policy.state_dict().items() if k not in expected})
    result=algo.policy.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and not set(result.missing_keys)&expected
    assert frozen_before==digest({k:v.cpu() for k,v in algo.policy.state_dict().items() if k not in expected})
    assert digest({k:v.cpu() for k,v in algo.policy.state_dict().items() if k in expected})==digest(state)
    algo.eval();torch.set_grad_enabled(False);model_before=digest({k:v.cpu() for k,v in algo.policy.state_dict().items()})
    factory=metric.SubprocVectorEnv;instances=[]
    def wrapped(*args,**kw):
        obj=Observer(factory(*args,**kw),cell,out,trace);instances.append(obj);return obj
    metric.SubprocVectorEnv=wrapped
    seed_all(cell['round_policy_seeds'][0])
    try:values=metric.evaluate_success(cfg,algo,bench,[cell['task']],None,see=seed)
    finally:metric.SubprocVectorEnv=factory
    rr=[r for x in instances for r in x.rounds];assert len(instances)==1 and len(rr)==2
    trials=[dict(point=cell['point'],seed=seed,task=cell['task'],arm=cell['arm'],trial=r['round']*10+i,round=r['round'],lane=i,success=int(v),first_success_step=r['first'][i],policy_steps=r['count']) for r in rr for i,v in enumerate(r['success'])]
    assert len(trials)==20 and abs(float(values[0])-sum(x['success'] for x in trials)/20)<1e-12
    assert model_before==digest({k:v.cpu() for k,v in algo.policy.state_dict().items()})
    task=bench.get_task(cell['task']);init=Path(cfg.init_states_folder)/task.problem_folder/task.init_states_file
    assert len(torch.load(init))>=20
    write(out/'TRIALS.json',trials);write(out/'ROUNDS.json',rr)
    write(out/'COMPLETE.json',{'status':'PASS','cell':cell,'successes':sum(x['success'] for x in trials),'trials':20,'seconds':time.time()-start,'checkpoint_sha256':rec['sha256'],'checkpoint_model_hash':rec['model_hash'],'full_eval_model_hash':model_before,'frozen_frontend_hash':frozen_before,'native_evaluator_sha256':sha(ROOT/'libero/lifelong/metric.py'),'init_file_sha256':sha(init),'initial_indices':list(range(20)),'engineering':tag,'files':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='COMPLETE.json'}})
    print('ROLLOUT_CELL_COMPLETE',index,cell,flush=True)
if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    run(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else None,not(len(sys.argv)>3 and sys.argv[3]=='no_trace'))
