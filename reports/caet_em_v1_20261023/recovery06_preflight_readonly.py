"""Stopped queue inspection: CPU strict state/teacher checks, no forwards or writes."""
import os,sys,json,subprocess,fcntl,time
from pathlib import Path
os.environ['LIBERO_CONFIG_PATH']='/data/mlr_ifa/config'
ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
from caet_em_exp.model import Policy,strict_policy_restore
from caet_em_exp.admission import equal
from caet_em_exp.dataset import diagnostic_dataset
contract=verify();assert contract=='2f6af444986a5710cbc84572f25ec15f9b3d44ea72aacbd01fa3ad4e624681b0'
assert not (ROOT/'controller.exit').exists()
processes=[]
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\0');cmd=(pp/'cmdline').read_bytes().decode(errors='replace')
  except OSError:continue
  if ('EM_ROOT='+str(ROOT)).encode() in env or str(ROOT) in cmd:processes.append(int(pp.name))
assert not processes,processes
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with open('/data/mlr_ifa/physical_effect_probe.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
root=arm_root('B2');recovery=root/'recovery/latest.pth';s=torch.load(recovery,map_location='cpu')
assert s['task']==8 and 0<s['next_epoch']<100 and s['cursor']==0
assert s['steps']==38400+s['next_epoch']*309 and s['contract']==contract
assert s['codec_sha256']==sha(frozen_codec()) and s['dataset_sha256']==sha(root/'dataset/task8.json')
assert s['teacher_sha256']==sha(root/'teachers/task8.pth')
assert equal(s['history'],js(root/'epochs/task8.json')) and len(s['history'])==s['next_epoch']
assert set(s['rng'])=={'python','numpy','cpu','cuda'} and set(s['grads'])==set(s['requires'])
assert s['optimizer']['state'] and s['scheduler']['last_epoch']==s['next_epoch']*309
pack=torch.load(frozen_codec(),map_location='cpu');p=Policy('B2',pack['model'],pack['statistics']);p.load_state_dict(s['model'],strict=True)
assert equal(p.state_dict(),s['model']) and set(s['modes'])==set(dict(p.named_modules()))
assert set(s['grads'])==set(dict(p.named_parameters()))
for n,v in p.named_parameters():
 assert v.shape==s['model'][n].shape and v.dtype==s['model'][n].dtype
 if s['grads'][n] is not None:assert s['grads'][n].shape==v.shape and torch.isfinite(s['grads'][n]).all()
assert all(torch.isfinite(v).all() for v in s['model'].values() if v.is_floating_point())
assert not (root/'models/task8.pth').exists() and not (root/'TASK_COMPLETE/task8.json').exists()
assert not (root/'models/task9.pth').exists() and not list(root.glob('evaluation/trials*.jsonl'))
teacher=js(root/'teachers/task8.json');assert teacher['sha256']==s['teacher_sha256']
assert teacher['binding']==dict(contract=contract,codec_sha256=sha(frozen_codec()),prior_policy_sha256=sha(root/'models/task7.pth'),dataset_sha256=s['dataset_sha256'])
cache=torch.load(root/'teachers/task8.pth',map_location='cpu');assert cache['binding']==teacher['binding'] and len(cache['z'])==3081
ds=diagnostic_dataset('B2',8);assert len(ds)==3081
checkpoints={}
for a,tasks in [('B0',range(6,10)),('B1',range(6,10)),('B2',(6,7))]:
 for t in tasks:
  cp=arm_root(a)/'models'/('task%d.pth'%t);raw=torch.load(cp,map_location='cpu');m=Policy(a,pack['model'],pack['statistics']);n=strict_policy_restore(m,raw,contract,sha(frozen_codec()))
  assert (ROOT/'phases'/('%s_train%d.exit'%(a,t))).read_text().strip()=='0'
  checkpoints['%s_task%d'%(a,t)]=dict(path=str(cp),sha256=sha(cp),strict_keys=n)
for a in ('B0','B1'):
 assert (ROOT/'phases'/(a+'_evaluate.exit')).read_text().strip()=='0' and (ROOT/'phases'/(a+'_audit.exit')).read_text().strip()=='0'
b1=torch.load(arm_root('B1')/'models/task6.pth',map_location='cpu');b2=torch.load(root/'models/task6.pth',map_location='cpu');assert equal(b1['model'],b2['model'])
disk=capacity();preserve_bytes=recovery.stat().st_size
# Existing private outputs are already allocated; reserve remaining B2 growth and a recovery copy.
stable=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.stat().st_nlink==1)
remaining=max(0,int(1.8*1024**3)-stable)
assert disk['/data']['root_free']-preserve_bytes-remaining>=2*1024**3
events=[json.loads(x) for x in (ROOT/'EVENTS.jsonl').read_text().splitlines()]
last=[x for x in events if x.get('event')=='main_epoch' and x.get('arm')=='B2'][-1]
assert last['epoch']==s['next_epoch']-1 and last['steps']==s['steps']
print(json.dumps(dict(status='PASS_STOPPED_EXACT_EPOCH_RECOVERABLE',observed_unix=time.time(),contract=contract,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uptime_seconds=float(Path('/proc/uptime').read_text().split()[0]),cause='UNDETERMINED_INTERRUPTION_NO_EXIT_MARKER',own_processes=processes,gpu_empty=True,lock_available=True,recovery=dict(path=str(recovery),sha256=sha(recovery),bytes=preserve_bytes,task=8,next_epoch=s['next_epoch'],steps=s['steps'],history_rows=len(s['history']),dataset_sha256=s['dataset_sha256'],teacher_sha256=s['teacher_sha256'],codec_sha256=s['codec_sha256'],full_fields_present=True,CPU_model_strict=True,GPU_exact_replay_pending=True),checkpoints=checkpoints,B1_B2_Task6_tensor_equal=True,formal_B2_episodes=0,dataset_identity_order_mask_payload_exact=True,last_epoch_event=last,disk=disk,stable_private_bytes=stable,remaining_B2_bound=remaining,data_after_preserve_remaining_and_reserve=disk['/data']['root_free']-preserve_bytes-remaining-2*1024**3,forward_calls=0,optimizer_calls=0,environment_steps=0,server_writes=0)))
