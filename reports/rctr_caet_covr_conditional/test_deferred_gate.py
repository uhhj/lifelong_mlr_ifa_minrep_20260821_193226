"""Pure scheduler checks: no optimizer, environment, or experiment launch."""
import copy
import json
from pathlib import Path
from branch_gate import next_action as original
from branch_gate_v3 import next_action

r=Path(__file__).resolve().parent
d=json.loads((r/'CAET_GOAL_DEFERRED.json').read_text())
def record(a='3/4',b='7/10'):
    return dict(engineering='PASS',auc=a,baseline_auc=b,**{k:'a'*64 for k in
        ('acceptance_sha256','baseline_acceptance_sha256','episodes_sha256','baseline_episodes_sha256')})
e={}
assert next_action(e,d)==original(e)
e={'RCTR':{'OBJECT':{'42':record('7/10','7/10')}}}
assert next_action(e,d)==original(e)
e['CAET']={'OBJECT':{str(s):record() for s in (41,42,43)}}
assert original(e)['suite']=='GOAL'
assert (next_action(e,d)['method'],next_action(e,d)['suite'],next_action(e,d)['seed'])==('COVR','OBJECT',42)
broken=copy.deepcopy(e);broken['CAET']['OBJECT']['43']['engineering']='FAIL'
assert next_action(broken,d)['action']=='BLOCKED'
missing=copy.deepcopy(e);del missing['CAET']['OBJECT']['43']
assert next_action(missing,d)['seed']==43 and next_action(missing,d)['method']=='CAET'
e['COVR']={'OBJECT':{'42':record()}}
assert next_action(e,d)['seed']==41
e['COVR']['OBJECT'].update({str(s):record() for s in (41,43)})
assert next_action(e,d)['method']=='COVR' and next_action(e,d)['suite']=='GOAL'
e['COVR']['GOAL']={str(s):record() for s in (41,42,43)}
assert next_action(e,d)['suite']=='LIBERO50'
e['RA_IFA']={'LIBERO50':{str(s):dict(engineering='PASS',acceptance_sha256='a'*64) for s in (41,42,43)}}
assert next_action(e,d)['action']=='RUNNABLE_ACCEPTED_WITH_DEFERRED'
assert 'CAET' not in {k for k in e['CAET'] if k=='GOAL'}
assert next_action(e,None)==original(e)
print(json.dumps(dict(status='PASS',checks=11,optimizer_calls=0,environment_steps=0,
    thresholds_unchanged=True,blocked_branch_never_marked_complete=True)))
