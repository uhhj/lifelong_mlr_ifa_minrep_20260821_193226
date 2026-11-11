"""Hold only the verified dispatcher for protocol review; children continue."""
from pathlib import Path
import os, signal, sys, json, time, hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0,str(Q))
from reverse_common import verify, processes, read
EXPECTED='5d28905059a9afc15cc8175097eee4ce3af5599154bb980bc54dce53d220795c'
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()==EXPECTED
verify(full=True)
pid=119466
proc=Path('/proc')/str(pid)
assert not (Q/'controller.exit').exists(), 'Controller already exited; no signal sent'
cmd=(proc/'cmdline').read_bytes().split(b'\0')
assert [x.decode() for x in cmd if x]==['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')]
assert proc.joinpath('cwd').resolve()==Q
before=processes()
status=read(Q/'STATUS.json')
assert status.get('status')=='PARTIAL_TRAINING', status
assert status.get('active'), status
receipt=Q/'DISPATCH_PROTOCOL_REVIEW_HOLD.json'
assert not receipt.exists(), 'Hold already recorded; no repeated mutation'
os.kill(pid,signal.SIGSTOP)
for unused in range(50):
    after=processes()
    own=[x for x in after if x['pid']==pid]
    if own and own[0]['state']=='T':break
    time.sleep(.02)
assert own and own[0]['state']=='T', 'Signal sent; verify state manually'
children=[x for x in after if x['pid']!=pid]
assert all(x['state'] not in ['T','t'] for x in children), children
data=dict(status='DISPATCH_HELD_FOR_PROTOCOL_REVIEW_ACTIVE_CHILDREN_CONTINUE',observed=time.time(),contract_sha256=EXPECTED,controller=pid,signal='SIGSTOP',before=before,after=after,coordinator_status_before=status,scientific_source_changes=0,optimizer_calls_added=0,environment_calls_added=0,child_signals_sent=0,note='Temporary hold of future dispatch only. No controller exit or experimental acceptance is claimed.')
with receipt.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
print(json.dumps(data))
