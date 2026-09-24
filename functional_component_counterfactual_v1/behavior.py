import os,sys,time,subprocess,traceback
from .common import *
# Spawn children must install the same historical LIBERO package binding before unpickling environments.
from cb_ifa_experiment.gpu_runtime import bind as bind_historical_runtime
bind_historical_runtime()
def cells():
 starts=read(OUT/'START_MANIFEST.json')['starts'];out=[]
 for s in starts:
  if not s['smoke']:continue
  for task in [s['task_id']-1,s['task_id']]:
   for arm in ['KEEP','GN_REMOVE']:
    out.append(dict(point=s['start_id'],seed=s['seed'],task=task,arm=arm,environment_seed=s['seed'],round_policy_seeds=[stable_seed('functional_component_counterfactual_v1/'+s['start_id']+'/'+str(task)+'/'+str(i)) for i in [0,1]]))
 return out
def run_cell(index):
 from cb_ifa_experiment.gpu_runtime import torch,np,seed_all,digest,eval_config
 import robomimic.utils.obs_utils as ObsUtils
 from libero.libero.benchmark import get_benchmark
 from libero.lifelong.algos import get_algo_class
 from libero.lifelong.utils import get_task_embs
 from libero.lifelong import metric
 from paper_qp_rollout_v1.evaluate import Observer
 import copy
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
 cell=cells()[index];dest=OUT/'behavior'/cell['point']/str(cell['task'])/cell['arm'];dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 cfg=eval_config(cell['seed'],dest);assert cfg.eval.n_eval==20
 seed_all(cell['seed']);ObsUtils.initialize_obs_utils_with_obs_specs({'obs':dict(cfg.data.obs.modality)})
 bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index);bench.set_task_embs(get_task_embs(cfg,[bench.get_task(i).language for i in range(bench.n_tasks)]))
 algo=get_algo_class(cfg.lifelong.algo)(bench.n_tasks,cfg).to('cuda:0')
 ck=torch.load(SCRATCH/(cell['point']+'_behavior.pt'),map_location='cpu');state=ck['native_model']
 expected={n for n in algo.policy.state_dict() if n.startswith(('temporal_transformer.','policy_head.'))};assert expected==set(state)
 result=algo.policy.load_state_dict(state,strict=False);assert not result.unexpected_keys and not(set(result.missing_keys)&expected)
 if cell['arm']=='GN_REMOVE':
  v=ck['remove_head'];offset=0
  with torch.no_grad():
   for p in algo.policy.policy_head.parameters():p.copy_(v[offset:offset+p.numel()].view_as(p));offset+=p.numel()
  assert offset==len(v)
 algo.eval();torch.set_grad_enabled(False);before=digest(algo.policy.state_dict())
 factory=metric.SubprocVectorEnv;instances=[]
 def wrapped(*a,**kw):
  obj=Observer(factory(*a,**kw),cell,dest,False);instances.append(obj);return obj
 metric.SubprocVectorEnv=wrapped;begin=time.monotonic()
 write(dest/'CONFIG.json',cfg)
 try:values=metric.evaluate_success(cfg,algo,bench,[cell['task']],None,see=cell['seed'])
 finally:metric.SubprocVectorEnv=factory
 rounds=[r for o in instances for r in o.rounds];assert len(rounds)==2
 trials=[dict(point=cell['point'],seed=cell['seed'],task=cell['task'],arm=cell['arm'],trial=r['round']*10+i,success=int(v),first_success_step=r['first'][i],policy_steps=r['count']) for r in rounds for i,v in enumerate(r['success'])]
 assert len(trials)==20 and abs(float(values[0])-sum(x['success'] for x in trials)/20)<1e-12
 assert before==digest(algo.policy.state_dict())
 write(dest/'ROUNDS.json',rounds);write(dest/'TRIALS.json',trials)
 write(dest/'COMPLETE.json',dict(status='PASS',cell=cell,trials=20,successes=sum(x['success'] for x in trials),seconds=time.monotonic()-begin,model_unchanged=True,formal_training_resumed=False))
def main():
 if len(list((OUT/'starts').glob('*/COMPLETE.json')))!=36:
  write(OUT/'BEHAVIORAL_REPORT.json',dict(status='NOT_RUN_OFFLINE_MEASUREMENT_INCOMPLETE',formal_training_resumed=False));return
 cc=cells();write(OUT/'BEHAVIOR_MANIFEST.json',dict(cells=cc,total_trials=480,source='six preselected smoke points, original get_action and horizon, same init indices0..19 and per-round RNG for paired arms'))
 for i,c in enumerate(cc):
  dest=OUT/'behavior'/c['point']/str(c['task'])/c['arm']
  if (dest/'COMPLETE.json').exists():continue
  status('BEHAVIOR',cell=i,total_cells=24,point=c['point'],task=c['task'],arm=c['arm'])
  log=OUT/('behavior_cell_%02d.log'%i)
  with log.open('w') as f:r=subprocess.run([sys.executable,'-u','-m','functional_component_counterfactual_v1.behavior',str(i)],stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:
   error=log.read_text()
   environmental=any(s in error for s in ['ImportError','ModuleNotFoundError','FileNotFoundError','EGL_NOT_INITIALIZED','Cannot initialize a EGL','Failed to initialize EGL','No module named'])
   write(OUT/'BEHAVIORAL_REPORT.json',dict(status='NOT_RUN_ENVIRONMENT_UNAVAILABLE' if environmental else 'FAILED_BEHAVIOR_EXECUTION',failed_cell=i,error=error[-6000:],completed_cells=i,formal_training_resumed=False))
   return
 trials=[];pairs=[]
 for c in cc:
  d=OUT/'behavior'/c['point']/str(c['task'])/c['arm'];trials.extend(read(d/'TRIALS.json'))
 for i in range(0,len(cc),2):
  a,b=cc[i:i+2];d=OUT/'behavior'/a['point']/str(a['task'])
  ar=read(d/'KEEP'/'ROUNDS.json');br=read(d/'GN_REMOVE'/'ROUNDS.json')
  assert [x['init_sha256'] for x in ar]==[x['init_sha256'] for x in br]
  assert [x['initial_obs_sha256'] for x in ar]==[x['initial_obs_sha256'] for x in br]
  aa=read(d/'KEEP'/'TRIALS.json');bb=read(d/'GN_REMOVE'/'TRIALS.json')
  for x,y in zip(aa,bb):pairs.append(dict(point=a['point'],seed=a['seed'],task=a['task'],trial=x['trial'],KEEP=x['success'],REMOVE=y['success'],difference=y['success']-x['success']))
 table(OUT/'BEHAVIOR_TRIALS.csv',trials);table(OUT/'BEHAVIOR_PAIRS.csv',pairs)
 import numpy as np
 rng=np.random.RandomState(2026092412);groups={}
 for p in pairs:groups.setdefault((p['point'],p['task']),[]).append(p['difference'])
 means=np.array([np.mean(v) for v in groups.values()]);boot=[np.mean([np.mean(rng.choice(v,len(v),replace=True)) for v in groups.values()]) for _ in range(5000)]
 write(OUT/'BEHAVIORAL_REPORT.json',dict(status='COMPLETE',trials=len(trials),paired_trials=len(pairs),remove_minus_keep=float(means.mean()),conditional_paired_trial_CI95=np.percentile(boot,[2.5,97.5]).tolist(),transition_counts={str(a)+str(b):sum(p['KEEP']==a and p['REMOVE']==b for p in pairs) for a in [0,1] for b in [0,1]},cell_values=[dict(point=k[0],task=k[1],difference=float(np.mean(v))) for k,v in groups.items()],interpretation='Low-powered20 trials/task, six correlated development states; no equivalence or long-term claim',formal_training_resumed=False))
if __name__=='__main__':
 import multiprocessing as mp
 mp.set_start_method('spawn',force=True);run_cell(int(sys.argv[1]))
