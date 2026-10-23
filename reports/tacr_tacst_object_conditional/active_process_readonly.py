"""Inspect actual active worker process ancestry and cwd; no model or server writes."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

Q = Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
rows = []
for p in Path('/proc').glob('[0-9]*'):
    try:
        cmd = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        cwd = (p/'cwd').resolve()
        if 'python -c' in cmd or not (str(Q) in cmd or cwd == Q or Q in cwd.parents):
            continue
        stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
        rows.append(dict(pid=int(p.name), ppid=int(stat[1]), start_ticks=int(stat[19]), cwd=str(cwd), command=cmd))
    except (OSError, ValueError, IndexError):
        pass
print(json.dumps(dict(status='READONLY_ACTUAL_PROCESS_SNAPSHOT', observed_unix=time.time(),
                      processes=rows,
                      gpu=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], universal_newlines=True),
                      main_epochs=json.loads((Path('/root/mlr_ifa/experiments/TACR_OBJECT_seed41_20261023')/'ra_coreset/epochs.json').read_text()),
                      frozen_manifest_sha256=hashlib.sha256((Q/'FROZEN_SHA256.json').read_bytes()).hexdigest() if (Q/'FROZEN_SHA256.json').exists() else None,
                      model_forwards=0, optimizer_updates=0, environment_steps=0, server_writes=0)))
