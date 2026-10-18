"""Wait outside GPU processes for the authorized predecessor; never restart it."""
import json,time,subprocess,shutil,hashlib
from pathlib import Path
Q=Path(__file__).resolve().parent
PRE=Path('/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017')
def write(x):
    p=Q/'RUN_STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(x,unix=time.time()),indent=2));tmp.replace(p)
def verify():
    for p,h in json.loads((Q/'ORCHESTRATION_SHA256.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    for seed in (42,43,44):
        W=Q/f'seed{seed}'
        for p,h in json.loads((W/'SOURCE_SHA256.json').read_text()).items():assert hashlib.sha256((W/'source'/p).read_bytes()).hexdigest()==h,p
verify()
while True:
    if (PRE/'queue.exit').exists():
        code=(PRE/'queue.exit').read_text().strip()
        if code!='0':
            write(dict(status='BLOCKED_PREDECESSOR',predecessor_exit=code))
            (Q/'BLOCKED.md').write_text('Previous OBJECT ZR40/41 queue failed. Preserve evidence, inspect predecessor; no GOAL GPU work started. After authorized recovery, resume only unstarted GOAL work, not whole frozen queue blindly.\n')
            raise SystemExit(72)
        ps=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        active=[x for x in ps if str(PRE) in x and ('run_queue.sh' in x or ' -m ' in x or 'supervise.sh' in x)]
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not active and not gpu:break
        write(dict(status='WAITING_GPU_RELEASE',gpu_pids=gpu,predecessor_processes=active))
    else:write(dict(status='WAITING_FOR_OBJECT_ZR40_41',predecessor=str(PRE)))
    time.sleep(120)
verify();free=shutil.disk_usage('/root').free
assert free>=12*1024**3,('Need >=12GiB available before GOAL',free)
write(dict(status='ADMITTED',free_system_bytes=free,predecessor_exit=0))
print('ADMITTED: previous queue exited 0, no predecessor/GPU compute process, sufficient system space')
