AUTHORIZATION={'predecessor_actual_final_status': 'PASS_COMPLETE_LOCAL_ACCEPTANCE', 'predecessor_final_audit_sha256': '95d642665d2f491288538c1f27f5188e895d012f9145b47eada8279a9b737a9f', 'predecessor_publication_status': 'PUSH_VERIFIED', 'predecessor_publication_commit': '493427977bda08b59a0c4cf43852292866d28b77', 'predecessor_remote_cpu_results': '/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1/final_cpu_engineering02_20261111_064213/CPU_RESULTS.json', 'predecessor_remote_cpu_results_sha256': '9e865093f4b0068c8a5dccb7198e7db046a207ae1820a7217c5a8c5a73be13ec', 'prepared_contract': '5d28905059a9afc15cc8175097eee4ce3af5599154bb980bc54dce53d220795c'}
"""Read-only proof that a pre-auth transport failure dispatched no scientific work."""
from pathlib import Path
import json,os,subprocess,sys,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0,str(Q))
from reverse_common import verify,capacity,processes,sha,read
assert sha(Q/'CONTRACT.json')==AUTHORIZATION['prepared_contract']
verify(full=True)
markers={n:(Q/n).exists() for n in ['LAUNCH.json','LAUNCH_AUTHORIZATION.json','controller.exit','controller.log']}
assert not any(markers.values()),('Possible earlier dispatch: do not duplicate',markers)
assert not processes(),'Possible reverse process: do not duplicate'
old=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
assert (old/'controller.exit').read_text().strip()=='0'
old_processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(str(p/'cwd'))
        if str(old) in cmd or cwd.startswith(str(old)):old_processes.append(dict(pid=int(p.name),cmd=cmd,cwd=cwd))
    except (OSError,ProcessLookupError,PermissionError):pass
assert not old_processes
cpu=Path(AUTHORIZATION['predecessor_remote_cpu_results']);assert sha(cpu)==AUTHORIZATION['predecessor_remote_cpu_results_sha256']
assert len(read(cpu)['cases'])==12 and all(x['status']=='PASS_CPU_CASE_RECONSTRUCTION' for x in read(cpu)['cases'].values())
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps(dict(status='PASS_NO_PRIOR_REVERSE_DISPATCH_AFTER_PREAUTH_FAILURE',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),markers=markers,reverse_processes=[],predecessor_processes=[],predecessor_actual_cpu_results_sha256=sha(cpu),capacity=capacity(),gpu=gpu.stdout,new_optimizer=0,new_environment=0,new_model_forward=0,server_writes=0)))
