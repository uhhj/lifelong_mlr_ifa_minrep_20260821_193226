"""Zero-training CPU-only temporal provenance audit; outputs metadata, never tensors."""
import os,json,time,hashlib,inspect
from pathlib import Path
import numpy as np
import torch
import h5py
from robomimic.utils.dataset import SequenceDataset
from robomimic.utils import obs_utils
obs_utils.initialize_obs_utils_with_obs_specs({'obs':{'low_dim':['joint_states','gripper_states']}})
torch.set_num_threads(1)
os.nice(10)
Q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
registry=js(Q/'REGISTRY.json');out=dict(started_unix=time.time(),seeds={},optimizer_calls=0,environment_calls=0,server_writes=0,model_forward_calls=0)
out['runtime_dataset_source']=inspect.getsource(SequenceDataset)
out['runtime_dataset_file']=inspect.getfile(SequenceDataset)
out['runtime_dataset_sha256']=sha(out['runtime_dataset_file'])
hdf_cache={}
for seed in (41,42,43):
 entry=registry[f'CAET_OBJECT_{seed}'];root=Path(entry['output']);source=Path(entry['source'])
 contract=js(root/'CONTRACT.json');binding=js(root/'PARENT_BINDING.json');parent=Path(binding['parent'])
 assert sha(parent/'CONTRACT.json')==binding['contract_sha256']
 assert sha(parent/'FEATURES.json')==binding['feature_manifest_sha256']==sha(root/'FEATURES.json')
 src={}
 for name in ('caet_exp/model.py','caet_exp/fit.py','caet_exp/observe.py','caet_exp/labels.py',
              'caet_exp/dataset.py','coreset_exp/model.py','coreset_exp/export.py',
              'libero/lifelong/datasets.py','libero/lifelong/models/base_policy.py',
              'libero/lifelong/models/complete_policy.py'):
  path=source/name;assert sha(path)==contract['source_sha256'][name]
  src[name]=dict(sha256=sha(path),text=path.read_text())
 tasks={}
 for task in (6,7,8,9):
  bp=root/'bindings'/f'task{task}.json';b=js(bp);hp=b['hdf5']
  if hp not in hdf_cache:hdf_cache[hp]=sha(hp)
  assert hdf_cache[hp]==b['hdf5_sha256']
  demos=sorted({x['demo_id'] for x in b['windows'].values() if x['identity_unique']})
  demos_report={};actions={}
  with h5py.File(hp,'r') as f:
   for demo in demos:
    g=f['data'][demo];state=g['states'][:];joint=g['obs/joint_states'][:];grip=g['obs/gripper_states'][:]
    actions[demo]=g['actions'][:].astype(np.float32)
    assert len(state)==len(joint)==len(grip)==len(actions[demo])
    post_j=np.array_equal(state[1:,1:8],joint[:-1]);post_g=np.array_equal(state[1:,8:10],grip[:-1])
    assert post_j and post_g
    pre_equal=np.all(state[:,1:8]==joint,axis=1)&np.all(state[:,8:10]==grip,axis=1)
    demos_report[demo]=dict(length=len(state),post_action_joint_exact=post_j,post_action_gripper_exact=post_g,
       pre_action_joint_gripper_equal_rows=int(pre_equal.sum()),different_from_pre_rows=int((~pre_equal).sum()),
       observation_keys=list(g['obs'].keys()))
  # Use the actual installed dataset implementation, without model or simulation.
  ds=SequenceDataset(hdf5_path=hp,obs_keys=['joint_states','gripper_states'],dataset_keys=['actions'],
      load_next_obs=False,frame_stack=1,seq_length=8,pad_frame_stack=True,pad_seq_length=True,
      get_pad_mask=False,goal_mode=None,hdf5_cache_mode=None,hdf5_use_swmr=False,
      hdf5_normalize_obs=False,filter_by_attribute=None)
  ds.load_demo_info(filter_by_attribute=None,demos=demos)
  checks=[]
  with h5py.File(hp,'r') as f:
   for demo in demos:
    length=len(actions[demo]);start=min(10,max(0,length-9));index=ds._demo_id_to_start_indices[demo]+start
    native=ds[index];idx=np.minimum(np.arange(start,start+8),length-1)
    assert np.array_equal(native['actions'],actions[demo][idx])
    # ObsUtils.process_obs performs the native float32 conversion. Compare exact
    # values after that declared conversion, not float64 file bytes to float32.
    for key in ('joint_states','gripper_states'):
     assert native['obs'][key].dtype==np.float32
     assert np.array_equal(native['obs'][key],f['data'][demo]['obs'][key][:][idx].astype(np.float32))
    checks.append(dict(demo=demo,start=start,target=int(idx[-1]),same_index_obs_actions=True,load_next_obs=False,observation_conversion='native exact float32'))
  valid=invalid=ambiguous=post_inputs=0
  for name,w in b['windows'].items():
   fp=root/'features'/str(task)/name;assert sha(fp)==w['sha256']
   v=torch.load(fp,map_location='cpu');a=v['actions'][0].numpy();assert v['feature'].shape==(1,8,4,512)
   if not w['identity_unique']:ambiguous+=1;continue
   starts=w['possible_starts'];raw=actions[w['demo_id']]
   assert starts and all(np.array_equal(a,raw[np.minimum(np.arange(st,st+8),len(raw)-1)]) for st in starts)
   targets={min(st+7,len(raw)-1) for st in starts}
   assert targets=={w['target']}
   if w['target']<len(raw)-1:
    valid+=1;post_inputs+=1
   else:invalid+=1
  la=js(root/'effects/current'/f'task{task}'/'AUDIT.json')
  assert la['counts']['valid']==valid
  tasks[str(task)]=dict(windows=len(b['windows']),valid_effect_windows=valid,invalid_terminal_windows=invalid,
     ambiguous_windows=ambiguous,valid_windows_with_last_input_post_target_action=post_inputs,
     cached_action_windows_and_feature_hashes='PASS',binding_sha256=sha(bp),hdf5_sha256=hdf_cache[hp],
     native_dataset_checks=checks,demos=demos_report,label_boundary=la['boundary'])
 out['seeds'][str(seed)]=dict(contract_sha256=sha(root/'CONTRACT.json'),parent=str(parent),
    feature_symlink_resolves_to_parent=(root/'features').resolve()==(parent/'features').resolve(),sources=src,tasks=tasks)
out.update(finished_unix=time.time(),cuda_initialized=torch.cuda.is_initialized(),status='AUDIT_EXECUTION_PASS',
 conclusion='POST_ACTION_INFORMATION_PRESENT: same-index observation includes action[t] outcome; effect input averages all 8 frames including target t.')
assert not out['cuda_initialized']
print('TEMPORAL_AUDIT_JSON='+json.dumps(out))
