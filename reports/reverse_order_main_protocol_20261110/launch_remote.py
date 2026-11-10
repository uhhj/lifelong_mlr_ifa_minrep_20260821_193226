AUTHORIZATION={'predecessor_actual_final_status': 'PASS_COMPLETE_LOCAL_ACCEPTANCE', 'predecessor_final_audit_sha256': '95d642665d2f491288538c1f27f5188e895d012f9145b47eada8279a9b737a9f', 'predecessor_publication_status': 'PUSH_VERIFIED', 'predecessor_publication_commit': '493427977bda08b59a0c4cf43852292866d28b77', 'predecessor_remote_cpu_results': '/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1/final_cpu_engineering02_20261111_064213/CPU_RESULTS.json', 'predecessor_remote_cpu_results_sha256': '9e865093f4b0068c8a5dccb7198e7db046a207ae1820a7217c5a8c5a73be13ec', 'prepared_contract': '5d28905059a9afc15cc8175097eee4ce3af5599154bb980bc54dce53d220795c'}
"""Unique authorized launch after predecessor actual final acceptance and publication."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0,str(Q))
from reverse_common import verify,capacity,read,write,environment,processes,PY,sha
assert AUTHORIZATION['predecessor_actual_final_status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert AUTHORIZATION['predecessor_publication_status']=='PUSH_VERIFIED'
assert AUTHORIZATION['prepared_contract']==sha(Q/'CONTRACT.json')
old=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
assert (old/'controller.exit').read_text().strip()=='0'
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(str(p/'cwd'))
        assert str(old) not in cmd and not cwd.startswith(str(old)),('Predecessor still has process',p.name)
    except (OSError,ProcessLookupError,PermissionError):pass
cpu=Path(AUTHORIZATION['predecessor_remote_cpu_results'])
assert sha(cpu)==AUTHORIZATION['predecessor_remote_cpu_results_sha256']
assert read(cpu)['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT'
assert len(read(cpu)['cases'])==12 and all(v['status']=='PASS_CPU_CASE_RECONSTRUCTION' for v in read(cpu)['cases'].values())
assert not processes() and not (Q/'LAUNCH.json').exists() and not (Q/'controller.exit').exists()
assert read(Q/'PREPARATION_COMPLETE.json')['status']=='PASS_PREPARED_FROZEN_NOT_LAUNCHED'
verify(full=True);space=capacity()
with (Q/'LAUNCH_AUTHORIZATION.json').open('x') as f:json.dump(AUTHORIZATION,f,indent=2)
with (Q/'controller.log').open('x') as log:
    p=subprocess.Popen([PY,'-B',str(Q/'controller.py')],cwd=Q,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(Q/'LAUNCH.json',dict(status='ACTUAL_CONTROLLER_DISPATCHED_NOT_EXPERIMENTAL_ACCEPTANCE',pid=p.pid,time=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),capacity=space,predecessor_final_audit=AUTHORIZATION['predecessor_final_audit_sha256'],predecessor_publication=AUTHORIZATION['predecessor_publication_commit']))
print(json.dumps(read(Q/'LAUNCH.json')))
