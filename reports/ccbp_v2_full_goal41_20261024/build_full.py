"""Local one-time derivation. Never rerun after source freeze/deployment."""
from pathlib import Path
import json,ast
R=Path(__file__).resolve().parent
P=R.parent/'20261024_ccbp_v2'
def put(n,s):
 if n.endswith('.py'):ast.parse(s)
 (R/n).write_text(s,encoding='utf8')
for n in ['losses.py','legacy_losses.py','prototype_loss.py','synthetic_checks.py','remote.py']:
 put(n,(P/n).read_text(encoding='utf8'))
put('publish.py',(P/'publish.py').read_text(encoding='utf8').replace('reports/ccbp_v2_goal41_20261024','reports/ccbp_v2_full_goal41_20261024'))
base=(P/'runtime.py').read_text(encoding='utf8')
base=base[:base.index('def capacity():')]+'''def capacity():
 out={}
 for mount,base,reserve,root_free in [('/root',ROOT_PRIVATE,1024**3,False),('/data',DATA_PRIVATE,2*1024**3,True)]:
  seen=set();allocated=0
  for p in base.rglob('*') if base.exists() else []:
   if not p.is_file() or p.is_symlink():continue
   st=p.stat();key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);allocated+=st.st_blocks*512
  v=os.statvfs(mount);free=(v.f_bfree if root_free else v.f_bavail)*v.f_frsize
  remaining=max(0,VOLUME_BOUND[mount]-allocated);assert free-remaining>=reserve,(mount,free,remaining)
  out[mount]=dict(free=free,unique_inode_allocated=allocated,remaining_bound=remaining,reserve=reserve)
 return out
class Dataset(torch.utils.data.Dataset):
 def __init__(self,stage):
  self.stage=stage;self.man=js(O/('ra_coreset/dataset_manifests/task%d.json'%stage))['rows'];assert len(self.man)==POP[stage]
  self.native=FileTrajectoryDataset([r['path'] for r in self.man]);labels={}
  for t in range(6,stage):
   p=O/('effects/memory/task%d.pth'%t);assert sha(p)==js(p.with_suffix('.json'))['sha256']
   labels.update({str(O/'ra_coreset/buffer'/n):v for n,v in torch.load(p,map_location='cpu').items()})
  p=O/('effects/current/task%d/labels.pth'%stage);assert sha(p)==js(p.parent/'AUDIT.json')['labels_sha256']
  labels.update({str(O/'features'/str(stage)/n):v for n,v in torch.load(p,map_location='cpu').items()})
  self.rows=[labels[r['path']] for r in self.man]
  for r,v in zip(self.man,self.rows):
   assert sha(r['path'])==r['sha256']==v['sample_sha256']
   assert all(r[k]==v[k] for k in ['task','demo','u','bc_valid'])
  self.ids=[i for i,r in enumerate(self.rows) if r['task']<stage and r['bc_valid']]
  assert sum(r['bc_valid'] for r in self.rows)==VALID[stage] and len(self.ids)==OLD[stage]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x,t=self.native[i];r=self.rows[i]
  x.update(next_action=torch.as_tensor(r['next_action'],dtype=torch.float32),causal_bc_valid=bool(r['bc_valid']),v3_index=i)
  return x,t
def make(arm):
 c=EasyDict(js(O/'resolved_ra_coreset.json'));c.experiment_dir=str(OUTPUT[arm]/'models');c.run_root=str(OUTPUT[arm]);c.buffer_dir=str(OUTPUT[arm]/'logical_buffer_empty')
 a,b=build(c);a.refs={};a.caet_epoch_statistics=[];a.disable_all=False
 assert a.step==0
 return a
def setup(a,ds):
 sampler=torch.utils.data.DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0)
 loader=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0,persistent_workers=False)
 a.start_task(-1,loader);a.refs={}
 for t in range(6,ds.stage+1):
  i=next(i for i,r in enumerate(ds.rows) if r['task']==t)
  f=ds.native[i][0]['feature'][0,:,0,:];a.refs[t]=f[0].cuda()
 return loader,sampler
'''+base[base.index('def capture(a):'):]
put('runtime.py',base)
put('layout.py',"""from pathlib import Path
ARMS='BCNV'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1')
OUTPUT={a:(ROOT_PRIVATE if a in 'BN' else DATA_PRIVATE)/a for a in ARMS}
VOLUME_BOUND={'/root':int(1.07*1024**3),'/data':int(1.88*1024**3)}
POP={6:997,7:1388,8:1848,9:3365}
VALID={6:917,7:1286,8:1723,9:3218}
OLD={6:0,7:477,8:899,9:1329}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return (ROOT_PRIVATE if (arm,stage)==('B',6) else DATA_PRIVATE)/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return DATA_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
""")
old=(P/'train.py').read_text(encoding='utf8')
update=old[old.index('def update('):old.index('def admit(')]
update=update.replace('info,measure=False','info,stage,measure=False').replace('tasks==9','tasks==stage').replace('k==9','k==stage').replace('a.refs[9]','a.refs[stage]').replace('tasks==8','tasks<stage')
update=update.replace("saved=rng();pars=[p for p in a.policy.parameters() if p.requires_grad];grad={};tic=time.time()","saved=rng();pars=[p for p in a.policy.parameters() if p.requires_grad];before_grads=[None if p.grad is None else p.grad.detach().clone() for p in pars];grad={};tic=time.time()")
update=update.replace("assert all(p.grad is None for p in pars) and equal(saved,rng())","assert all((p.grad is None if g is None else p.grad is not None and torch.equal(p.grad,g)) for p,g in zip(pars,before_grads)) and equal(saved,rng())")
put('updates.py','from runtime import *\nfrom losses import aux\nfrom legacy_losses import aux as legacy_aux\n'+update)
# Reuse exact feature-cached paired evaluator, expanded to all retained stages.
ev=(P/'evaluate.py').read_text(encoding='utf8')
ev=ev.replace('Exactly 320','Exactly 800').replace('def load(arm,c,shape):','def load(arm,stage,c,shape):').replace("OUTPUT[arm]/'stage9.pth'","model_path(arm,stage)")
ev=ev.replace("load('B',c,shape)","load('B',6,c,shape)").replace('budget=320','budget=800')
start=ev.index(' for arm in ARMS:');end=ev.index(' assert len(records)==320')
loop=ev[start:end].splitlines()
loop=[loop[0], '  for stage in range(6,10):']+[' '+line for line in loop[1:]]
block='\n'.join(loop)+'\n'
block=block.replace('load(arm,c,shape)','load(arm,stage,c,shape)').replace('for taskid in [6,7,8,9]:','for taskid in range(6,stage+1):').replace("run='%s_t%d_g%d'%(arm,taskid,group)","run='%s_s%d_t%d_g%d'%(arm,stage,taskid,group)").replace('dict(arm=arm,task=taskid','dict(arm=arm,stage=stage,task=taskid')
ev=ev[:start]+block+ev[end:]
ev=ev.replace('len(records)==320','len(records)==800').replace('episodes=320,vectors=32,cells=16','episodes=800,vectors=80,cells=40')
put('evaluate.py',ev)
check=(P/'check_readonly.py').read_text(encoding='utf8').replace('CCBP_V2_GOAL41_20261024_v1','CCBP_V2_FULL_GOAL41_20261024_v1')
put('check_readonly.py',check)
print('Local derivation complete; not deployed or admitted')

if __name__=='__main__':pass
