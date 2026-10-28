"""Read-only GPU availability and interrupted checkpoint metadata; no CUDA init or updates."""
import json, hashlib, time
from pathlib import Path
import torch
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1')
P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1/OBJECT_41/S')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=json.loads((Q/'CONTRACT.json').read_text())
assert all(sha(Path(p))==h for p,h in contract['files'].items())
p=P/'recovery.pth'; c=torch.load(p,map_location='cpu')
h=json.loads((Q/'OBJECT_41/EPOCHS_S.json').read_text())
assert c['history']==h and c['next_epoch']==len(h)==7
assert c['state']['step']==h[-1]['steps']==71834
assert c['physical_optimizer_calls']==2534
assert set(c['state'])=={'model','optimizer','scheduler','grads','modes','requires','rng','step'}
b=json.loads((Q/'OBJECT_41/BINDING.json').read_text())
assert c['origin_sha256']==sha(Path(b['checkpoint']))
assert c['probe_sha256']==sha(Q/'OBJECT_41/PROBES.json')
pci=[]
for x in Path('/sys/bus/pci/devices').glob('*'):
    if (x/'vendor').read_text().strip()=='0x10de':
        pci.append(dict(address=x.name,device=(x/'device').read_text().strip(),driver=str((x/'driver').resolve()) if (x/'driver').exists() else None))
print(json.dumps(dict(status='READONLY_INTERRUPTION_CHECK_NOT_RECOVERY_ACCEPTANCE',observed=time.time(),checkpoint=dict(sha256=sha(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime,next_epoch=c['next_epoch'],step=c['state']['step'],state_keys=sorted(c['state']),rng_keys=sorted(c['state']['rng']),history_exact=True,origin_exact=True,probe_exact=True,contract=c['contract'],physical_optimizer_calls=c['physical_optimizer_calls']),output_inventory=[dict(name=x.name,bytes=x.stat().st_size,mtime=x.stat().st_mtime) for x in P.iterdir() if x.is_file()],pci_nvidia=pci,device_nodes=[x.name for x in Path('/dev').glob('nvidia*')],nvidia_modules=[x for x in Path('/proc/modules').read_text().splitlines() if x.startswith('nvidia')],driver_version=Path('/proc/driver/nvidia/version').read_text() if Path('/proc/driver/nvidia/version').exists() else None,train_status=json.loads((Q/'OBJECT_41/TRAIN_STATUS.json').read_text()),controller_exit_exists=(Q/'controller.exit').exists(),train_exit_exists=(Q/'OBJECT_41/train_S.exit').exists(),new_optimizer_calls=0,new_environments=0)))
