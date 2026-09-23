import collections,gzip,subprocess,sys,time,shutil
import numpy as np,torch,h5py
from .common import *

def main():
 assert not (OUT/'PROTOCOL.json').exists(),'Frozen preparation already exists'
 from cb_ifa_experiment import gpu_runtime as g
 from libero.libero.benchmark import get_benchmark
 with gzip.open(ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz','rt') as f:design=json.load(f)
 origins={}
 for seed in (45,46,47):
  for row in lines(ROOT/f'reports/functional_geometry_diagnostic_v1/carrier/seed{seed}/origins.jsonl'):origins[row['point']]=row
 starts=[];probes=[];assets=[];catalog={};plans={};replay_steps=0
 for seed in (45,46,47):
  cfg=g.train_config(seed,SCRATCH/'bootstrap');bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
  action_maps={}
  for task in (6,7):
   raw=ROOT/'data'/bench.get_task_demonstration(task);idx={}
   with h5py.File(raw,'r') as f:
    for demo in sorted(f['data'],key=lambda s:int(s.split('_')[-1]))[:10]:
     a=np.asarray(f['data'][demo]['actions'],dtype=np.float32)
     for t in range(len(a)):
      key=np.ascontiguousarray(a[np.minimum(np.arange(t,t+8),len(a)-1)]).tobytes()
      idx.setdefault(key,[]).append((demo,t,len(a)))
   action_maps[task]=idx
  for task in (7,8):
   spec=design['source_plan_sha256'][f'{seed}/{task}'];path=ROOT/spec['path'];assert sha(path)==spec['sha256']
   plan=read(path);plans[f'{seed}/{task}']=spec;pool=[];identity={}
   for i,origin in enumerate(plan['immutable_feature_origins']):
    t=int(Path(origin).parent.name)
    if t>=task:continue
    if origin not in catalog:
     z=torch.load(ROOT/origin,map_location='cpu');matches=action_maps[t].get(np.ascontiguousarray(z['actions'].numpy().reshape(8,7),dtype=np.float32).tobytes(),[])
     assert sha(ROOT/origin)==plan['record_sha256'][i]
     catalog[origin]=dict(matches=matches,sha256=plan['record_sha256'][i])
    matches=catalog[origin]['matches'];identity[i]=matches
    if len(matches)==1:
     demo,w,length=matches[0];pool.append(dict(task_id=t,trajectory_id=demo,window_index=w,trajectory_length=length,record_id=i,origin=origin,sha256=catalog[origin]['sha256']))
   points=sorted([p for p in design['points'] if p['seed']==seed and p['task']==task],key=lambda p:p['stage_step']);assert len(points)==20
   prev=None
   for p in points:
    row=origins[p['point']];anchors=[o for o in origins.values() if o['seed']==seed and o['task']==task and o['anchor'] and o['stage_step']<=p['stage_step']]
    anchor=max(anchors,key=lambda o:o['stage_step']);ap=Path(anchor['anchor']['path'])
    if str(ap) not in [a['path'] for a in assets]:
     assert sha(ap)==anchor['anchor']['sha256'];s=torch.load(ap,map_location='cpu')
     required=['model','optimizer','scheduler','rng','training_modes','optimizer_step','scheduler_step'];assert all(k in s for k in required)
     assets.append(dict(path=str(ap),sha256=sha(ap),bytes=ap.stat().st_size,complete=True,point=s['point']['point'],optimizer_step=s['optimizer_step'],scheduler_step=s['scheduler_step']));del s
    prev=max(prev if prev is not None else 0,anchor['stage_step']);replay_steps+=p['stage_step']-prev;prev=p['stage_step']
    batch=p['future20_record_indices'][0];used=set(batch);old_tasks=list(range(6,task));chosen=[];overlaps=[]
    for t in old_tasks:
     forbidden={m[0] for i in batch for m in identity.get(i,[]) if int(Path(plan['immutable_feature_origins'][i]).parent.name)==t}
     eligible=[r for r in pool if r['task_id']==t and r['record_id'] not in used and r['trajectory_id'] not in forbidden]
     # Require whole-trajectory disjointness from this update, not merely distinct windows.
     grouped=collections.defaultdict(list)
     for r in eligible:grouped[r['trajectory_id']].append(r)
     assert grouped,(p['point'],t,'no disjoint probe trajectory')
     local=np.random.RandomState(int(hashlib.sha256((p['point']+'/'+str(t)).encode()).hexdigest()[:8],16))
     demos=sorted(grouped);local.shuffle(demos)
     for d in demos:local.shuffle(grouped[d])
     count=24//len(old_tasks)
     for j in range(count):
      d=demos[j%len(demos)];assert grouped[d];r=dict(grouped[d].pop(),start_id=p['point']);chosen.append(r)
     overlaps.append(dict(task=t,excluded_batch_trajectories=sorted(forbidden),probe_trajectories=sorted({r['trajectory_id'] for r in chosen if r['task_id']==t})))
    assert len(chosen)==24
    record=dict(start_id=p['point'],seed=seed,transition=f'{task-1}->{task}',task_id=task,stage=p['phase'],epoch=p['epoch'],batch_index=p['batch'],stage_step=p['stage_step'],global_step=p['stage_step']+sum(read(ROOT/f'runs/fixed_kappa_confirmation_v2/formal/seed{seed}/mlr/plans/task{t}.json')['batches_per_epoch']*100 for t in range(6,task)),optimizer_step=p['stage_step'],scheduler_step=p['stage_step'],plan=spec,anchor=anchor,expected_witness=row['witness'],record_indices=batch,new_batch_id=[plan['immutable_feature_origins'][i] for i in batch if int(Path(plan['immutable_feature_origins'][i]).parent.name)==task],replay_batch_id=[plan['immutable_feature_origins'][i] for i in batch if int(Path(plan['immutable_feature_origins'][i]).parent.name)<task],probe_isolation=overlaps)
    starts.append(record);probes.extend(chosen)
 context=dict(git_branch=subprocess.check_output(['git','branch','--show-current']).decode().strip(),git_HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),git_status=subprocess.check_output(['git','status','--short']).decode(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']).decode().strip(),formal_status=read(ROOT/'reports/sample_qp_lifelong_v1_continuation_v4/STATUS.json'),formal_training_resumed=False,source_cohort='new_native_3090_functional_geometry_v1',restorable_anchors=assets,policy_head='GMMHead; learned scale; original terminal GMM NLL',trainable_blocks=['temporal decoder','policy head'],disk_free=dict(data=shutil.disk_usage('/data').free,root=shutil.disk_usage('/root').free),limitations=['development seeds','epochs0-29 unavailable; only30-99; no early-phase inference','task8->9 full distributed origins unavailable; use complete6->7 and7->8','historical model-only checkpoints excluded','origins on common carrier remain serially dependent; starts are not independent windows'])
 write(OUT/'EXPERIMENT_CONTEXT.json',context)
 write(OUT/'START_MANIFEST.json',dict(starts=starts,count=len(starts),minimal_native_reconstruction_steps=replay_steps,plans=plans,selection='All pre-existing outcome-independent120 origins, no filtering on observed conflict or fidelity'))
 write(OUT/'PROBE_MANIFEST.json',dict(windows=probes,count=len(probes),current_batch_trajectory_disjoint=True,history_unseen=False,selection='old retained buffer origins, exact8-action lineage, ambiguous excluded before loss; round-robin trajectory-balanced with private fixed RNG'))
 # Fixed engineering interpretation; abstain on sparse/no harm cases.
 protocol=dict(version=1,alphas=ALPHAS,parameter_scope='all formal trainable decoder+head float32 parameters',formal_training_resumed=False,seed_role='development',candidate='native MLR backward/clip/AdamW/scheduler inside isolated snapshot, restore afterwards',raw_gradient='same clipped MLR gradient, equal Euclidean norm',probe='eval all modules, learned GMM variance, terminal expert NLL; no sampled actions',deadband='tau_i=max(2*repeat_abs_difference,32*float32_eps*max(1,abs(base_NLL_i))); prediction uses same units and tau_i; frozen before directional outcomes',statistics_unit='start then equal seed-transition strata; report neutral coverage and undefined metrics',reliability=dict(median_spearman=.8,median_sign_accuracy=.85,median_harm_recall=.8,median_harm_precision=.8,minimum_defined_start_fraction=.5,minimum_decisive_window_fraction=.5,all_six_strata_required=True),classification='alpha1 reliable A; .5/.25 reliable B; .125 reliable B with explicit1/8 extension; <=.0625 reliable C; else D; engineering rules not theorem',hvp_trigger='alpha1 unreliable AND smallest-alpha median Spearman improves>=.15 AND sign accuracy improves>=.10; require decisive coverage>=.5 at smaller alpha',hvp_starts='first middle/late point per seed-transition (12), plus last middle point each stratum(6), plus last late in first2 sortedstrata(2);20 total',hvp_probes='first3 frozen windows per oldtask (3 or6); true NLL double-autograd HVP, no Fisher',normal_path_restore='exact source witness for all120 starts; no new carrier on mismatch',quantization='record norm(actual float32 written displacement-alpha*d), h dot actual displacement separately; primary remains alpha*h dot d',raw_comparison='two self-direction fidelity curves do not prove relative same-target superiority; additionally wrong-direction proxy h dot d_raw against actual AdamW y',execution='reconstruct only missing origins from nearest saved anchor/previous restored origin; no formal checkpoints or solver called',implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
 write(OUT/'PROTOCOL.json',protocol);print('PREPARED',len(starts),len(probes),replay_steps,flush=True)
if __name__=='__main__':main()
