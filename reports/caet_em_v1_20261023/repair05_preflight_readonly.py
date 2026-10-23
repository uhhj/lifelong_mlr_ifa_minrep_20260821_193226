"""Stopped-queue repair evidence. CPU only; no forward/update/environment/write."""
import os,sys,json,hashlib,random,fcntl,subprocess
from pathlib import Path
os.environ['LIBERO_CONFIG_PATH']='/data/mlr_ifa/config'
ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
from caet_em_exp.model import Policy,strict_policy_restore
assert ROOT.name.endswith('engineering04')
contract=verify();assert contract=='58e5a91bad19cc3e2c17e0ecc7d2fd430666b88150cf34f92ce4ecad09c94c94'
assert (ROOT/'controller.exit').read_text().strip()=='1'
assert (ROOT/'phases/B0_train7.exit').read_text().strip()=='0'
assert (ROOT/'phases/B0_diagnose7.exit').read_text().strip()=='1'
assert "KeyError: 't7__1360.pt'" in (ROOT/'phases/B0_diagnose7.log').read_text()
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\0')
  except OSError:continue
  assert ('EM_ROOT='+str(ROOT)).encode() not in env
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with open('/data/mlr_ifa/physical_effect_probe.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
root=arm_root('B0');pack=torch.load(frozen_codec(),map_location='cpu');checks={}
for t in (6,7):
 cp=root/'models'/('task%d.pth'%t);raw=torch.load(cp,map_location='cpu');p=Policy('B0',pack['model'],pack['statistics']);n=strict_policy_restore(p,raw,contract,sha(frozen_codec()))
 epochs=js(root/'epochs'/('task%d.json'%t));assert len(epochs)==100 and epochs[-1]['steps']==sum(BATCHES[j]*100 for j in range(6,t+1))
 manifest=js(root/'dataset'/('task%d.json'%t));paths=sorted(str(root/'buffer'/r['destination']) for j in range(6,t) for r in js(PARENT/'ra_coreset/storage'/('task%d.json'%j))['selected'])+sorted(str(p) for p in (PARENT/'features'/str(t)).glob('*.pt'));random.Random(41).shuffle(paths)
 assert len(paths)==len(manifest['rows'])==POPS[t]
 for path,row in zip(paths,manifest['rows']):
  name=Path(path).name;tid=int(name.split('__')[0][1:]) if '__' in name else t;sample=name.split('__',1)[1] if '__' in name else name
  assert (tid,sample,sha(path))==(row['task'],row['sample'],row['sha256'])
 storage=js(root/'storage'/('task%d.json'%t));assert storage['selected']==js(PARENT/'ra_coreset/storage'/('task%d.json'%t))['selected']
 side=root/'sidecars'/('task%d.pth'%t);md=js(side.with_suffix('.json'));assert sha(side)==md['sha256']==storage['sidecar_sha256']
 for item in storage['selected']:assert sha(root/'buffer'/item['destination'])==item['sha256']
 checks[str(t)]=dict(strict_keys=n,checkpoint_sha256=sha(cp),epochs=100,steps=raw['steps'],dataset_sha256=sha(root/'dataset'/('task%d.json'%t)),exact_training_population=True,sidecar_sha256=sha(side))
recovery=root/'recovery/latest.pth';s=torch.load(recovery,map_location='cpu');cp7=torch.load(root/'models/task7.pth',map_location='cpu')
assert s['next_epoch']==100 and s['task']==7 and s['steps']==38400 and s['cursor']==0
assert s['contract']==contract and s['codec_sha256']==sha(frozen_codec()) and s['dataset_sha256']==checks['7']['dataset_sha256'] and s['teacher_sha256'] is None
assert s['model'].keys()==cp7['model'].keys() and all(torch.equal(s['model'][k],cp7['model'][k]) for k in s['model'])
assert set(s['rng'])=={'python','numpy','cpu','cuda'} and set(s['grads'])==set(s['requires']) and s['optimizer']['state'] and s['scheduler'] and len(s['history'])==100
assert not (root/'diagnostics/task7.json').exists() and not (root/'models/task8.pth').exists()
assert all(not list(arm_root(a).rglob('*.pth')) for a in ('B1','B2'))
assert not list(root.glob('evaluation/trials*.jsonl'))
disk=capacity();additional_preserved_recovery=recovery.stat().st_size
assert disk['/data']['root_free']-int(4.1*1024**3)-additional_preserved_recovery>0
assert disk['/']['ordinary_free']-int(4.6*1024**3)>0
print(json.dumps(dict(status='PASS_STOPPED_COMPLETE_STAGE_REUSE',contract=contract,checkpoints=checks,recovery=dict(sha256=sha(recovery),bytes=additional_preserved_recovery,complete_epoch=100,task=7,steps=38400,model_equals_task7=True,full_fields_present=True,actual_GPU_restore_pending=True),buffer_members=len(list((root/'buffer').glob('*.pt'))),diagnostic_expected_population=2261,incorrect_post_retain_extra_task7_members=740,codec_sha256=sha(frozen_codec()),disk=disk,forward_calls=0,optimizer_calls=0,environment_steps=0,server_writes=0)))
