"""User-requested stop: preserve every artifact; prevent controller advancing first."""
import os,signal,time,json,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1')
P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'CONTRACT.json')=='9b53df5a8fbe894e7b27887462b7b25b4e8477a9ad12820deb8df162632fb978'
c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(Path(p))==h for p,h in c['files'].items())
def processes():
    out={}
    for p in Path('/proc').glob('[0-9]*'):
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeError):continue
        if str(Q) in cmd and int(p.name)!=os.getpid():out[int(p.name)]=cmd
    return out
E=Q/'user_stop_20261028';E.mkdir(exist_ok=False)
before=processes();actions=[]
(E/'REQUEST.json').write_text(json.dumps(dict(time=time.time(),reason='User explicitly stops remaining mechanism experiments and prioritizes full latest S OBJECT seeds 41 42 43',processes=before),indent=2))
for pid,cmd in before.items():
    if 'controller' in cmd:
        assert processes().get(pid)==cmd;os.kill(pid,signal.SIGTERM);actions.append([pid,'SIGTERM_CONTROLLER'])
time.sleep(1)
for pid,cmd in processes().items():
    if 'torch.distributed.run' in cmd:
        os.kill(pid,signal.SIGINT);actions.append([pid,'SIGINT_LAUNCHER'])
for _ in range(30):
    if not processes():break
    time.sleep(1)
for pid,cmd in processes().items():
    os.kill(pid,signal.SIGTERM);actions.append([pid,'SIGTERM_REMAINING'])
for _ in range(10):
    if not processes():break
    time.sleep(1)
assert not processes(),processes()
assert all(sha(Path(p))==h for p,h in c['files'].items())
preserved={};epochs={}
for p in Q.rglob('EPOCHS_*.json'):
    name=p.relative_to(Q).as_posix();epochs[name]=len(json.loads(p.read_text()));preserved[name]=sha(p)
checkpoints={str(p.relative_to(P)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in P.rglob('recovery.pth')}
out=dict(status='USER_STOPPED_PARTIAL_PRESERVED_NOT_SCIENTIFIC_ACCEPTANCE',observed=time.time(),actions=actions,processes_after=processes(),frozen_files=56,epochs=epochs,epoch_sha256=preserved,recovery_checkpoints=checkpoints,unlogged_partial_epoch_cost='UNKNOWN',deleted_files=0,scientific_parameters_changed=False)
(E/'RESULT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
