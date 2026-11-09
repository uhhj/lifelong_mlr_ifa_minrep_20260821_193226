SPEC={'OBJECT_41': {'parent': '/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc', 'align_parent': '/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01', 'align_source': '/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source'}, 'OBJECT_42': {'parent': '/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_OBJECT_seed42_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source'}, 'OBJECT_43': {'parent': '/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_OBJECT_seed43_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_43/source'}, 'GOAL_41': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source'}, 'GOAL_42': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed42_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_42/source'}, 'GOAL_43': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed43_20261017', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed43_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_43/source'}}

from pathlib import Path
import os,json,hashlib,time,random,math
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.nice(10)
import torch,numpy as np,h5py
torch.set_num_threads(1)
def js(p):return json.loads(Path(p).read_text())
cache={}
def sha(p):
 p=str(Path(p).resolve());s=Path(p).stat();key=(p,s.st_size,s.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
out={};files={}
for case,s in SPEC.items():
 P=Path(s['parent']);A=Path(s['align_parent']);seed=int(case[-2:]);features=js(P/'FEATURES.json')['hashes'];rows={};labelsmeta={};storages={};stage_manifests={};before={};source_files={}
 def bind(path):
  path=str(path);h=sha(path);source_files[path]=h;files[path]=h;return h
 for t in range(6,10):
  bp=A/'bindings'/('task%d.json'%t);binding=js(bp);bind(bp);hp=binding['hdf5'];assert bind(hp)==binding['hdf5_sha256']
  lp=A/'effects/current'/('task%d'%t)/'labels.pth';aud=js(lp.with_name('AUDIT.json'));bind(lp.with_name('AUDIT.json'));assert bind(lp)==aud['labels_sha256']
  assert sha(bp)==aud['binding_sha256'];labels=torch.load(lp,map_location='cpu');arrays={}
  with h5py.File(hp,'r') as h:
   for demo in sorted({w['demo_id'] for w in binding['windows'].values() if w.get('identity_unique',True)}):
    g=h['data'][demo];actions=g['actions'][:].astype(np.float32);states=g['states'][:];joint=g['obs/joint_states'][:];grip=g['obs/gripper_states'][:]
    assert np.array_equal(states[1:,1:8],joint[:-1]) and np.array_equal(states[1:,8:10],grip[:-1]);arrays[demo]=actions
  current=sorted((P/'features'/str(t)).glob('*.pt'));assert len(current)==len(binding['windows'])==len(labels)
  for f in current:
   w=binding['windows'][f.name];l=labels[f.name];fh=bind(f);assert fh==features[str(t)+'/'+f.name]==w['sha256']==l['sample_sha256']
   unique=w.get('identity_unique',True);starts=w['possible_starts'];n=w['length'];u=w['target']
   valid=bool(unique and starts and all(st>=0 and st+7<n and st+7==u for st in starts) and u+1<n)
   assert valid==bool(l['bc_valid']),(case,t,f.name,valid,l['bc_valid'])
   if unique:
    actions=arrays[w['demo_id']];assert len(actions)==n
    payload=torch.load(f,map_location='cpu');assert payload['feature'].shape==(1,8,4,512)
    assert all(np.array_equal(payload['actions'][0].numpy(),actions[np.minimum(np.arange(st,st+8),n-1)]) for st in starts)
   if valid:assert l['demo']==w['demo_id'] and l['u']==u and np.array_equal(np.asarray(l['next_action']),arrays[w['demo_id']][u+1])
   rows[(t,f.name)]=dict(task=t,sample=f.name,sha256=fh,bc_valid=valid,demo=l.get('demo'),u=l.get('u'))
  labelsmeta[str(t)]=dict(path=str(lp),sha256=sha(lp),binding_path=str(bp),binding_sha256=sha(bp),hdf5=hp,hdf5_sha256=sha(hp),valid=sum(rows[(t,f.name)]['bc_valid'] for f in current),total=len(current))
  sp=P/'random/storage'/('task%d.json'%t);old=js(sp);bind(sp);assert old['before']==before
  selected=random.Random(seed).sample(current,max(1,math.ceil(.5*len(current))))
  assert [f.name for f in selected]==[Path(r['source']).name for r in old['selected']]
  mix=[]
  for name,h in sorted(before.items()):
   j=int(name.split('__')[0][1:]);sample=name.split('__',1)[1];r=dict(rows[(j,sample)],relative_path='random/buffer/'+name);assert r['sha256']==h;mix.append(r)
  mix += [dict(rows[(t,f.name)],relative_path='features/'+str(t)+'/'+f.name) for f in current]
  random.Random(seed).shuffle(mix);expected_batches=math.ceil(len(mix)/10)
  stage_manifests[str(t)]=dict(stage=t,population=len(mix),batches=expected_batches,bc_valid=sum(r['bc_valid'] for r in mix),rows=mix)
  for f,r in zip(selected,old['selected']):assert sha(f)==r['sha256'];before[r['destination']]=r['sha256']
  storages[str(t)]=old
 c=js(P/'resolved_random.json');bind(c['pretrain']);initial=js(P/'random/INITIAL_STATE.json');bind(P/'random/INITIAL_STATE.json')
 for name in ['CONTRACT.json','MODAL_CONTRACT.json','FEATURES.json','resolved_random.json','resolved_eval.json','random/TRAINING_COMPLETE.json']:bind(P/name)
 out[case]=dict(status='PASS_CPU_RANDOM_ALIGN_INPUTS',labels=labelsmeta,manifests=stage_manifests,storage=storages,initial_state=initial,files=source_files,production_budget=sum(v['batches']*100 for v in stage_manifests.values()),baseline_gpu=s.get('baseline_gpu'),all_random_members_and_order_exact=True,raw_hdf5_next_actions_and_full_window_mask_exact=True)
assert sum(r['production_budget'] for r in out.values())==544800
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_SIX_CASE_RANDOM_ALIGN_INPUT_AUDIT',observed=time.time(),cases=out,new_optimizer=0,new_environments=0,new_gpu_calls=0,full_validation='feature bytes, raw next actions, boundary masks, original Random membership/order, per-stage population',free_bytes={p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in ['/','/data']})))
