"""Audit the final full-epoch recovery state, without forwards, CUDA calls or writes."""
import os,sys,json,random,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/READOUT_CAPACITY_OBJECT41_20261023_v1_engineering02')
os.environ.update(PBD_ROOT=str(Q),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');sys.path.insert(0,str(Q))
from pbd.common import *
from pbd.models import build
contract=verify();assert (Q/'controller.exit').read_text().strip()=='0'
assert all(p.read_text().strip()=='0' for p in Q.glob('*.exit'))
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\0')
  except OSError:continue
  assert ('PBD_ROOT='+str(Q)).encode() not in env
p=OUT/'private/recovery.pth';h=sha(p);r=torch.load(p,map_location='cpu')
required={'version','key','epoch','next_epoch','model','optimizer','best_model','best_epoch','best_score','selected','updates','torch_rng','cuda_rng','numpy_rng','python_rng','modes','requires_grad','grads','contract_sha256','patch_manifest_sha256'}
assert set(r)==required and r['key']=='S_task7_fold4' and r['epoch']==199 and r['next_epoch']==200 and r['updates']==1600
assert r['contract_sha256']==contract and r['patch_manifest_sha256']==sha(V2/'output/PATCH_MANIFEST.json')
model=build('S');model.load_state_dict(r['model'],strict=True);assert sum(v.numel() for v in model.parameters())==400775
assert set(r['requires_grad'])==set(r['grads'])==set(dict(model.named_parameters()))
assert set(r['modes'])==set(dict(model.named_modules()))
for n,v in model.named_parameters():
 assert isinstance(r['requires_grad'][n],bool);v.requires_grad_(r['requires_grad'][n]);g=r['grads'][n]
 assert g is None or g.shape==v.shape and torch.isfinite(g).all();v.grad=g
for n,v in model.named_modules():assert isinstance(r['modes'][n],bool);v.training=r['modes'][n]
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);opt.load_state_dict(r['optimizer'])
for group in opt.param_groups:
 assert group['lr']==3e-4 and group['weight_decay']==.01
 for v in group['params']:
  st=opt.state[v];assert st and float(st['step'])==1600
  for k in ('exp_avg','exp_avg_sq'):assert st[k].shape==v.shape and torch.isfinite(st[k]).all()
assert all(torch.isfinite(v).all() for v in r['model'].values())
torch.set_rng_state(r['torch_rng']);np.random.set_state(r['numpy_rng']);random.setstate(r['python_rng'])
assert isinstance(r['cuda_rng'],list) and len(r['cuda_rng'])==1 and all(v.dtype==torch.uint8 and v.ndim==1 and v.numel()>0 for v in r['cuda_rng'])
rows=[json.loads(l) for l in (OUT/'EVENTS.jsonl').read_text().splitlines()];epochs=[v for v in rows if v['event']=='epoch' and v['key']==r['key']]
assert [v['epoch'] for v in epochs]==list(range(200));best=min(epochs,key=lambda v:v['validation_position_m'])
assert r['best_epoch']==best['epoch'] and r['best_score']==best['validation_position_m']
for budget in (50,100,200):
 s=r['selected'][budget];case=js(OUT/'cases'/(r['key']+'_b%d.json'%budget));assert s==case['selection'] and sha(s['path'])==s['sha256']
bestcp=torch.load(r['selected'][200]['path'],map_location='cpu');assert r['best_model'].keys()==bestcp['model'].keys() and all(torch.equal(v,bestcp['model'][n]) for n,v in r['best_model'].items())
assert sha(p)==h and verify()==contract
disk={mount:dict(root=(s:=os.statvfs(mount)).f_bfree*s.f_frsize,ordinary=s.f_bavail*s.f_frsize) for mount in ('/data','/')}
assert disk['/data']['root']>=2*1024**3 and disk['/']['ordinary']>=1024**3
print(json.dumps(dict(status='PASS_COMPLETE_RECOVERY_BINDING_AND_CLOSURE',contract_sha256=contract,recovery_sha256=h,key=r['key'],epoch=r['epoch'],next_epoch=r['next_epoch'],optimizer_steps=r['updates'],model_parameters=400775,optimizer_parameter_states=len(opt.state),full_state_fields=sorted(required),best_epoch=r['best_epoch'],best_score=r['best_score'],best_cp_sha256=r['selected'][200]['sha256'],phase_exit_sha256={v.name:sha(v) for v in Q.glob('*.exit')},all_phase_exit0=True,no_own_process=True,disk=disk,model_forwards=0,optimizer_calls=0,cuda_calls=0,environment_steps=0,server_writes=0)))
