"""Recheck cancellation, retained checkpoints/logs and zero queue processes."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023').resolve()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

marker=Q/'USER_TERMINATION_BY_USER.json'
saved=json.loads(marker.read_text())
assert saved['status']=='USER_CANCELLED_ALL_TACR_TACRST_STOP_CONFIRMED'
residual=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        cwd=(p/'cwd').resolve()
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if int(p.name)!=os.getpid() and stat[0]!='Z' and (cwd==Q or Q in cwd.parents):
            residual.append(int(p.name))
    except (OSError,ValueError):
        pass
assert not residual
retained=[]
logs=[]
for key,row in saved['entries'].items():
    for cp in row.get('checkpoints',[]):
        p=Path(cp['path'])
        assert p.is_file() and p.stat().st_size==cp['bytes'] and sha(p)==cp['sha256']
        retained.append(cp)
    O=Path(row['output'])
    for p in [*O.glob('*.log'),*(Q/key/'admission').glob('*.log'),*Q.glob(key+'.log')]:
        if p.is_file():
            logs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
c=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
assert all(sha(p)==h for p,h in c['files'].items())
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],universal_newlines=True)
print(json.dumps(dict(status='READONLY_USER_STOP_AND_RETAINED_ARTIFACTS_VERIFIED',observed_unix=time.time(),
    marker_sha256=sha(marker),residual_queue_processes=residual,gpu=gpu,gpu_idle=not gpu.strip(),
    retained_checkpoint_files=len(retained),all_checkpoint_sizes_sha256_match=True,logs=logs,
    frozen_files=len(c['files']),frozen_hashes_unchanged=True,server_writes=0,
    model_forwards=0,optimizer_updates=0,environment_steps=0)))
