"""Remove the agent-imposed dispatcher hold; preserve original science/queue."""
from pathlib import Path
import hashlib,json,os,signal,sys,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0,str(Q))
from reverse_common import verify,processes,read
EXPECTED='5d28905059a9afc15cc8175097eee4ce3af5599154bb980bc54dce53d220795c'
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()==EXPECTED
verify()
pid=119466;proc=Path('/proc')/str(pid)
assert not (Q/'controller.exit').exists() and not (Q/'FAILURE.json').exists()
assert proc.joinpath('cwd').resolve()==Q
assert [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')]
assert (proc/'stat').read_text().split(') ',1)[1].split()[0]=='T'
hold=read(Q/'DISPATCH_PROTOCOL_REVIEW_HOLD.json');assert hold['controller']==pid and hold['contract_sha256']==EXPECTED
target=Q/'DISPATCH_RESUMED_ORIGINAL_AUTHORIZATION.json'
assert not target.exists(),'Resume already recorded; do not repeat mutation'
before=processes();state=read(Q/'STATUS.json');assert state['status']=='PARTIAL_TRAINING' and not state['failures']
kernel_exits=[]
for name,child in state['active'].items():
    p=Path('/proc')/str(child)
    if p.exists():
        fields=(p/'stat').read_text().split(') ',1)[1].split()
        if fields[0]=='Z':
            assert len(fields)>=50 and int(fields[49])==0,(name,fields[49])
            kernel_exits.append(dict(case=name,pid=child,state='Z',raw_exit_code=int(fields[49])))
    failure=Q/'cases'/name/'FAILURE.json';assert not failure.exists(),str(failure)
tic=time.time();os.kill(pid,signal.SIGCONT)
for unused in range(50):
    actual=(proc/'stat').read_text().split(') ',1)[1].split()[0]
    if actual not in ['T','t']:break
    time.sleep(.05)
assert actual not in ['T','t'],'SIGCONT sent; review actual state before any further action'
data=dict(status='ORIGINAL_AUTHORIZED_DISPATCH_RESUMED',observed=time.time(),signal_time=tic,controller=pid,controller_state=actual,contract_sha256=EXPECTED,before=before,after=processes(),coordinator_before=state,kernel_completed_child_exit_observations=kernel_exits,scientific_source_changes=0,new_controller_launched=False,completed_training_restarted=False,child_signals_sent=0,note='Only SIGCONT of the existing verified dispatcher. Removes the agent-imposed review hold; existing finite twelve-run authorization and frozen method-specific main-table protocols remain in force. Remaining authorized queue now proceeds normally.')
with target.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
print(json.dumps(data))
