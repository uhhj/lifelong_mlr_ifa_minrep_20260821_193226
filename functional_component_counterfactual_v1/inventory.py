import collections,itertools,subprocess,sys,shutil,importlib
import numpy as np,torch,h5py
from .common import *
def spread(rows,n):
 rows=sorted(rows,key=lambda r:(r['window_index'],r['record_id']))
 if len(rows)<n:return None
 return [rows[int(i)] for i in np.linspace(0,len(rows)-1,n).round().astype(int)]
def choose(pool,demos,n):
 parts=[spread(pool[d],n//len(demos)+(i<n%len(demos))) for i,d in enumerate(demos)]
 return None if any(x is None for x in parts) else [r for part in parts for r in part]
def assign(pool,used,old):
 # Deterministic identities only: T first outside U, then G/D from other trajectories.
 demos=sorted(pool,key=lambda d:int(d.split('_')[-1]))
 for td in itertools.combinations([d for d in demos if d not in used],4):
  tr=choose(pool,td,128)
  if tr is None:continue
  rem=[d for d in demos if d not in td]
  if old:
   for gd in itertools.combinations(rem,4):
    gr=choose(pool,gd,128)
    if gr is None:continue
    for dd in itertools.combinations([d for d in rem if d not in gd],2):
     dr=choose(pool,dd,64)
     if dr is not None:return dict(G=gr,D=dr,T=tr)
  else:
   for dd in itertools.combinations(rem,2):
    dr=choose(pool,dd,64)
    if dr is not None:return dict(D=dr,T=tr)
 return None
def main():
 if (OUT/'INVENTORY_COMPLETE.json').exists():return read(OUT/'INVENTORY_COMPLETE.json')
 from cb_ifa_experiment import gpu_runtime as g
 from libero.libero.benchmark import get_benchmark
 source=read(SOURCE/'START_MANIFEST.json');starts=[]
 for seed in [45,46,47]:
  for task in [7,8]:
   group=sorted([s for s in source['starts'] if s['seed']==seed and s['task_id']==task],key=lambda s:(s['epoch'],s['batch_index'],s['global_step']))
   ids=[0,4,8,11,15,19] if len(group)==20 else np.linspace(0,len(group)-1,6).round().astype(int).tolist()
   assert len(group)>=6
   starts.extend(dict(group[i],selection_index=i,cell_order=j,smoke=j==2) for j,i in enumerate(ids))
 write(OUT/'START_MANIFEST.json',dict(source=str(SOURCE/'START_MANIFEST.json'),source_sha256=sha(SOURCE/'START_MANIFEST.json'),starts=starts,count=36))
 files=['actual_update_fidelity_v1/engine.py','actual_update_fidelity_v1/run.py','gradient_relation_experiment/kernel.py','cb_ifa_experiment/gpu_runtime.py','gn_functional_geometry_v1/engine.py','gn_h3_subspace_closure_v1/project.py','libero/lifelong/models/base_policy.py','libero/lifelong/models/brief.py','libero/lifelong/models/policy_head.py']
 context=dict(head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),branch=subprocess.check_output(['git','branch','--show-current']).decode().strip(),tracked_diff=subprocess.check_output(['git','status','--short','-uno']).decode(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader']).decode().strip(),disk_free={p:shutil.disk_usage(p).free for p in ['/data','/root']},formal_training_resumed=False,sources={p:dict(path=str(ROOT/p),sha256=sha(ROOT/p),tracked=bool(subprocess.check_output(['git','ls-files',p]).strip())) for p in files},imports={n:importlib.import_module(n).__file__ for n in ['cb_ifa_experiment.gpu_runtime','gn_functional_geometry_v1.engine','gn_h3_subspace_closure_v1.project','actual_update_fidelity_v1.engine']})
 write(OUT/'EXPERIMENT_CONTEXT.json',context)
 (OUT/'SOURCE_AND_CONTEXT.md').write_text('# Source and context\n\n'+json.dumps(context,indent=2)+'\n')
 lineage={};roles={};availability=[];raw_maps={}
 for seed in [45,46,47]:
  cfg=g.train_config(seed,SCRATCH/'bootstrap');bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
  for task in [7,8]:
   group=[s for s in starts if s['seed']==seed and s['task_id']==task]
   plan=read(ROOT/group[0]['plan']['path']);assert sha(ROOT/group[0]['plan']['path'])==group[0]['plan']['sha256']
   pools=collections.defaultdict(lambda:collections.defaultdict(list));raws={}
   for t in range(6,task+1):
    path=ROOT/'data'/bench.get_task_demonstration(t)
    if str(path) not in raw_maps:
     ix={};raw={}
     with h5py.File(path,'r') as f:
      for demo in sorted(f['data'],key=lambda x:int(x.split('_')[-1]))[:10]:
       a=np.asarray(f['data'][demo]['actions'],np.float32);raw[demo]=a
       for w in range(len(a)):ix.setdefault(np.ascontiguousarray(a[np.minimum(np.arange(w,w+8),len(a)-1)]).tobytes(),[]).append((demo,w,len(a)))
     raw_maps[str(path)]=(ix,raw)
    raws[t]=(path,)+raw_maps[str(path)]
   allrows=[]
   for i,(o,h) in enumerate(zip(plan['immutable_feature_origins'],plan['record_sha256'])):
    t=int(Path(o).parent.name);path,ix,raw=raws[t]
    if o not in lineage:
     z=torch.load(ROOT/o,map_location='cpu');matches=ix.get(np.ascontiguousarray(z['actions'].numpy().reshape(8,7),np.float32).tobytes(),[])
     assert matches,('MISSING_LINEAGE',o);lineage[o]=matches
    ms=lineage[o];row=dict(record_id=i,origin=o,sha256=h,task_id=t,trajectory_candidates=ms)
    if len(ms)==1:
     demo,w,length=ms[0];actions=raw[demo];term=min(w+7,length-1)
     changes=np.flatnonzero(np.diff(np.sign(actions[:,6]))!=0)+1
     row.update(trajectory_id=demo,window_index=w,trajectory_length=length,terminal_index=term,time_stratum=['early','middle','late'][min(2,3*term//length)],gripper_change=bool(np.any(abs(changes-term)<=2)))
     pools[t][demo].append(row)
    allrows.append(row)
   for s in group:
    uid=[allrows[i] for i in s['record_indices']]
    used={t:{v[0] for r in uid if r['task_id']==t for v in r['trajectory_candidates']} for t in range(6,task+1)}
    selected={};errors=[]
    for t,role in [(task-1,'old'),(task,'new')]:
     a=assign(pools[t],used[t],role=='old')
     if a is None and role=='new':
      # Existing evaluator-only windows from demos10-49, never exposed to G/D/U.
      members=read(ROOT/'reports/paired_continuation/isolated_prepared_v1/MEMBERSHIP.json')['membership'][str(t)]
      ext=[]
      for bank in ['A','B']:
       for q in members[bank]:
        with h5py.File(ROOT/q['hdf'],'r') as hf:act=np.asarray(hf['data'][q['demo']]['actions'])
        terminal=min(q['start']+7,len(act)-1);changes=np.flatnonzero(np.diff(np.sign(act[:,6]))!=0)+1
        ext.append(dict(origin=q['path'],sha256=sha(ROOT/q['path']),record_id=None,task_id=t,trajectory_id=q['demo'],window_index=q['start'],trajectory_length=len(act),terminal_index=terminal,time_stratum=['early','middle','late'][min(2,3*terminal//len(act))],gripper_change=bool(np.any(abs(changes-terminal)<=2)),evaluation_privilege=True,source_monitor_bank=bank))
      td=sorted(d for d in pools[t] if d not in used[t])
      for dd in itertools.combinations(sorted(d for d in pools[t] if d not in td),2):
       dr=choose(pools[t],dd,64);tr=choose(pools[t],td,128-len(ext)) if td else None
       if dr is not None and tr is not None:a=dict(D=dr,T=tr+ext);break
     if a is None:errors.append(dict(task=t,reason='INSUFFICIENT_TRAJECTORY_SEPARATION_OR_WINDOWS'))
     else:
      for k,rows in a.items():selected[k+'_'+role]=rows
    for t in range(6,task-1):
     candidates=[d for d in pools[t] if d not in used[t]]
     rr=choose(pools[t],candidates[:min(4,len(candidates))],16) if candidates else None
     selected['T_monitor_'+str(t)]=rr or []
    checked={}
    for side in ['old','new']:
     rr={k.split('_')[0]:v for k,v in selected.items() if k.endswith('_'+side)}
     if not rr:continue
     sets={k:{r['trajectory_id'] for r in v} for k,v in rr.items()}
     checked[side]=all(not (sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)) and not (sets['T']&used[task-1 if side=='old' else task])
    assert all(checked.values())
    roles[s['start_id']]=dict(U=uid,roles=selected,checks=checked,status='READY' if not errors else 'INSUFFICIENT_TRAJECTORY_SEPARATION',errors=errors,task_names={str(t):bench.get_task(t).language for t in range(6,task+1)})
    availability.append(dict(start_id=s['start_id'],old_demo_counts={d:len(r) for d,r in pools[task-1].items()},new_demo_counts={d:len(r) for d,r in pools[task].items()},U_trajectories={str(t):sorted(v) for t,v in used.items()},status=roles[s['start_id']]['status'],counts={k:len(v) for k,v in selected.items()}))
   print('INVENTORIED',seed,task,flush=True)
 write(OUT/'DATA_ROLES.json',dict(starts=roles,availability=availability,history_unseen=False,selection='lexicographic feasible trajectory assignment T outside U then G then D; evenly spaced windows; identity-only; retained stage plan throughout',T_privileged_raw_access=any(r.get('evaluation_privilege',False) for s in roles.values() for rows in s['roles'].values() for r in rows)))
 protocol=dict(name='functional_component_counterfactual_v1',user_protocol_sha256=sha(OUT/'USER_PROTOCOL_v1.md'),head=context['head'],start_manifest_sha256=sha(OUT/'START_MANIFEST.json'),data_roles_sha256=sha(OUT/'DATA_ROLES.json'),rank=16,spectrum_rank=32,eigensolver=read(ROOT/'reports/gn_functional_geometry_v1/PROTOCOL.json')['eigensolver'],numerics=read(ROOT/'reports/gn_functional_geometry_v1/PROTOCOL.json')['numerics'],random_seeds=[2026092401,2026092402,2026092403],lambdas=[1.,.5,0.],kl_samples=64,kl_seed=2026092410,bootstrap_replicates=5000,bootstrap_seed=2026092411,block_length=2,paired_comparison_interval=.975,numerical_floor=1e-7,equivalence_floor=1e-4,tie_rule='within tau_num/2 weighted old+new numerical errors; prefer KEEP then HALF then REMOVE',selection='0.5 trajectory-equal old NLL+0.5 trajectory-equal new NLL on D only; freeze before opening T',rollouts=dict(points=[s['start_id'] for s in starts if s['smoke']],arms=['KEEP','GN_REMOVE'],trials_per_task=20,task_roles=['old','new'],total=480),formal_training_resumed=False,QP_called=False,physical_QP_gates_used=False,basis_retention='sequential temporary basis; retain spectrum, selected coefficients, projected vectors for smoke endpoints, seeds, operator source and data hashes; reproducible rebuilding',protocol_status='FROZEN_BEFORE_D_T_OUTCOMES')
 write(OUT/'PROTOCOL.json',protocol)
 done=dict(status='PASS' if all(a['status']=='READY' for a in availability) else 'PARTIAL_DATA_AVAILABILITY',ready=sum(a['status']=='READY' for a in availability),total=36,formal_training_resumed=False)
 write(OUT/'INVENTORY_COMPLETE.json',done);status('INVENTORY_COMPLETE',**done)
 return done
