"""Run only after all finite queue processes have exited. CPU, no forward/update/env."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['LIBERO_CONFIG_PATH']='/data/mlr_ifa/config'
ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
from caet_em_exp.model import Codec,Policy,strict_policy_restore
assert (ROOT/'controller.exit').read_text().strip()=='0';contract=verify()
recovery06=None
if (ROOT/'RECOVERY06_BINDING.json').exists():
 binding=js(ROOT/'RECOVERY06_BINDING.json');q=Path(binding['recovery_root'])
 assert sha(q/'EXECUTION_CONTRACT.json')==binding['execution_contract_sha256'] and sha(q/'RECOVERY_MANIFEST.json')==binding['manifest_sha256']
 assert binding['science_contract']==contract and all(sha(p)==h for p,h in js(q/'EXECUTION_CONTRACT.json')['files'].items())
 assert (q/'controller.exit').read_text().strip()=='0'
 admission=js(q/'RECOVERY_ADMISSION.json');resolution=js(ROOT/'RECOVERY06_RESOLUTION.json')
 assert admission['status']=='PASS' and admission['exact_next_batch_replay'] and admission['isolated_optimizer_calls']==2
 assert admission==js(ROOT/'RECOVERY06_ADMISSION.json') and admission['science_contract']==contract
 assert resolution['status']=='PASS_COMPLETE_EPOCH_CONTINUATION' and resolution['repeated_completed_epochs']==0 and resolution['admission_sha256']==sha(q/'RECOVERY_ADMISSION.json')
 assert js(arm_root('B2')/'epochs/task8.json')[:90]==js(q/'provenance/task8_epochs.json')
 assert js(ROOT/'ADMISSION_B2.json')['isolated_optimizer_calls']==3 and (ROOT/'phases/B2_admission_recovery06.exit').read_text().strip()=='0'
 recovery06=dict(binding=binding,admission=admission,resolution=resolution,preserved_complete_epoch_prefix_exact=True,extra_isolated_optimizer_calls=5,extra_setup_tokenizer_windows=6162,interrupted_partial_epoch_work='Unknown; not included as completed scientific updates',server_recovery_closed=True)
lineage=None
if (ROOT/'RESUME_MANIFEST.json').exists():
 lineage=js(ROOT/'RESUME_MANIFEST.json');admission=js(ROOT/'CONTINUATION_ADMISSION.json')
 assert admission['status']=='PASS' and admission['contract']==contract and admission['actual_GPU_full_state_roundtrip']
 assert admission['steps']==38400 and not admission['retrain_completed']
 assert sha(lineage['preserved_recovery']['path'])==lineage['preserved_recovery']['sha256']==admission['recovery_sha256']
 assert js(ROOT/'ADMISSION_B0.json')['contract']==contract
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
print(json.dumps(dict(status='PASS',contract=contract,engineering_lineage=lineage,recovery06=recovery06,checkpoints=reports,codec_sha256=sha(frozen_codec()),codec_selected_epoch=best,B1_B2_Task6_exact=True,initial=initial,forward_calls=0,optimizer_calls=0,environment_steps=0,server_writes=0)))
