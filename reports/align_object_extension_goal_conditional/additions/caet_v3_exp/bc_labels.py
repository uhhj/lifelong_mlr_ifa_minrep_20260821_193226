"""CPU-only identity/next-action labels; no physical-effect assumptions or model."""
import hashlib,json,collections
from pathlib import Path
import h5py,numpy as np,torch

def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def derive(parent,task):
 P=Path(parent);binding=js(P/'bindings'/('task%d.json'%task));hp=binding['hdf5']
 assert sha(hp)==binding['hdf5_sha256']
 arrays={};alignment={}
 with h5py.File(hp,'r') as h:
  for demo in sorted({w['demo_id'] for w in binding['windows'].values() if w.get('identity_unique',True)}):
   g=h['data'][demo];actions=g['actions'][:].astype(np.float32);states=g['states'][:]
   assert len(actions)==len(states) and actions.shape[1:]==(7,) and np.isfinite(actions).all()
   joint=g['obs/joint_states'][:];grip=g['obs/gripper_states'][:]
   assert np.array_equal(states[1:,1:8],joint[:-1]) and np.array_equal(states[1:,8:10],grip[:-1]),('post-action observation identity',task,demo)
   arrays[demo]=actions;alignment[demo]=dict(length=len(actions),post_observation_alignment_exact=True)
 rows={};proof=[];invalid=collections.Counter()
 for name,w in binding['windows'].items():
  path=P/'features'/str(task)/name;assert sha(path)==w['sha256']
  unique=w.get('identity_unique',True)
  if not unique:invalid['AMBIGUOUS_IDENTITY']+=1;continue
  actions=arrays[w['demo_id']];n=len(actions);u=w['target'];starts=w['possible_starts'];assert starts and n==w['length']
  payload=torch.load(path,map_location='cpu');assert payload['feature'].shape==(1,8,4,512)
  assert all(np.array_equal(payload['actions'][0].numpy(),actions[np.minimum(np.arange(st,st+8),n-1)]) for st in starts)
  if any(st<0 or st+7>=n for st in starts):invalid['PADDED_WINDOW']+=1;continue
  if u+1>=n:invalid['NO_NEXT_ACTION']+=1;continue
  assert all(st+7==u for st in starts)
  a=actions[u+1].copy();ah=hashlib.sha256(a.tobytes()).hexdigest()
  rows[name]=dict(task=task,sample=name,demo=w['demo_id'],u=u,bc_valid=True,next_action=a,sample_sha256=w['sha256'])
  proof.append(dict(sample=name,task=task,demo=w['demo_id'],u=u,next_action_index=u+1,max_input_observation_index=u,
                    next_action_sha256=ah,sample_sha256=w['sha256'],possible_starts=starts))
 return rows,dict(status='PASS',task=task,total_windows=len(binding['windows']),valid_bc_labels=len(rows),invalid=dict(invalid),
                  alignment=alignment,labels_metadata=proof,binding_sha256=sha(P/'bindings'/('task%d.json'%task)),hdf5_sha256=binding['hdf5_sha256'])
