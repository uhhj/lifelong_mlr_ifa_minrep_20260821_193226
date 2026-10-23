"""Run only after all finite queue processes have exited. CPU, no forward/update/env."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
from caet_em_exp.model import Codec,Policy,strict_policy_restore
assert (ROOT/'controller.exit').read_text().strip()=='0';contract=verify()
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\0')
  except OSError:continue
  assert ('EM_ROOT='+str(ROOT)).encode() not in env,'Queue still has a process'
pack=torch.load(frozen_codec(),map_location='cpu');codec=Codec(torch.load(ROOT/'shared/normalization.pth',map_location='cpu'));codec.load_state_dict(pack['model'],strict=True)
history=js(ROOT/'CODEC_TRAINING_AND_VALIDATION.json')['epochs'];assert len(history)==100;best=min(range(100),key=lambda i:history[i]['validation_loss']);assert best==pack['selected_epoch']
reports={};states={}
from libero.libero.benchmark import get_benchmark
benchmark=get_benchmark('LIBERO_OBJECT')(0);c=config();initial={}
for t in range(6,10):
 task=benchmark.get_task(t);ip=Path(c.init_states_folder)/task.problem_folder/task.init_states_file;arr=torch.load(ip,map_location='cpu');initial[str(t)]=dict(file_sha256=sha(ip),initial_hashes=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in arr[:20]])
for arm in ARMS:
 root=arm_root(arm);reports[arm]={}
 for t in range(6,10):
  manifest=js(root/'dataset'/('task%d.json'%t));history=js(root/'epochs'/('task%d.json'%t));n=len(manifest['rows'])
  oldpaths=sorted([str(root/'buffer'/r['destination']) for j in range(6,t) for r in js(PARENT/'ra_coreset/storage'/('task%d.json'%j))['selected']]);paths=oldpaths+sorted(str(p) for p in (PARENT/'features'/str(t)).glob('*.pt'));random.Random(41).shuffle(paths)
  assert len(paths)==n
  for path,row in zip(paths,manifest['rows']):
   name=Path(path).name;tid=int(name.split('__')[0][1:]) if '__' in name else t;sample=name.split('__',1)[1] if '__' in name else name
   assert tid==row['task'] and sample==row['sample'] and sha(path)==row['sha256'],'Native population/mix identity mismatch'
  for row in history:
   generator=torch.Generator();generator.manual_seed(row['epoch']);order=torch.randperm(n,generator=generator).numpy().astype(np.int64)
   assert hashlib.sha256(order.tobytes()).hexdigest()==row['order_sha256'],'Native sampler0 epoch order mismatch'
  p=Policy(arm,pack['model'],pack['statistics']);cp=root/'models'/('task%d.pth'%t);raw=torch.load(cp,map_location='cpu');sd=p.state_dict();assert set(sd)==set(raw['model'])
  assert all(v.shape==raw['model'][k].shape and v.dtype==raw['model'][k].dtype==torch.float32 for k,v in sd.items() if v.is_floating_point())
  n=strict_policy_restore(p,raw,contract,sha(frozen_codec()));reports[arm][str(t)]=dict(sha256=sha(cp),strict_keys=n,parameters=sum(v.numel() for v in p.parameters()),dtype_shape_exact=True)
  if t==6 and arm in ('B1','B2'):states[arm]=raw['model']
  del p
 recovery=torch.load(root/'recovery/latest.pth',map_location='cpu');assert recovery['next_epoch']==100 and recovery['task']==9 and recovery['steps']==105500
 assert recovery['contract']==contract and recovery['codec_sha256']==sha(frozen_codec());assert set(recovery['grads'])==set(recovery['requires']);assert set(recovery['rng'])=={'python','numpy','cpu','cuda'}
 reports[arm]['recovery']=dict(sha256=sha(root/'recovery/latest.pth'),steps=recovery['steps'],optimizer_states=len(recovery['optimizer']['state']),complete_epoch=True)
assert states['B1'].keys()==states['B2'].keys() and all(torch.equal(states['B1'][k],states['B2'][k]) for k in states['B1']),'B1/B2 Task6 must coincide before T/F starts'
print(json.dumps(dict(status='PASS',contract=contract,checkpoints=reports,codec_sha256=sha(frozen_codec()),codec_selected_epoch=best,B1_B2_Task6_exact=True,initial=initial,forward_calls=0,optimizer_calls=0,environment_steps=0,server_writes=0)))
