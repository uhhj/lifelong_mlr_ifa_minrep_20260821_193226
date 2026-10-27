"""Prepare local finite fork sources; no remote operations."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;E=R.parent
P=E/'20261027_c_goal42_43_object41_42_43'
for n in ['remote.py','credential_store.py','credential_session.py','queue_remote.py']:
    (R/n).write_text((P/n).read_text(encoding='utf8'),encoding='utf8')
s=(P/'publish.py').read_text(encoding='utf8').replace('reports/c_goal42_43_object41_42_43_20261027','reports/support_gate_forks_20261027').replace('C GOAL42-43 OBJECT41-43 fixed-protocol extension','Support-calibrated retention finite common-start forks')
(R/'publish.py').write_text(s,encoding='utf8')
cases={}
for suite,seed,stage in [('GOAL',41,8),('GOAL',42,8),('GOAL',43,8),('OBJECT',41,9)]:
    name='%s_%d'%(suite,seed)
    if suite=='GOAL' and seed==41:
        local=E/'20261025_ccbp_v4_full';root=json.loads((local/'ACTIVE_ROOT.json').read_text(encoding='utf8'))['root'];src=root;private='/data/mlr_ifa/experiments/CCBP_V4_FULL_GOAL41_20261025_v1/S'
    else:
        local=E/('20261026_ccbp_v4_s_goal42_43' if suite=='GOAL' else '20261026_ccbp_v4_s_object41_42_43')
        root=json.loads((local/'ACTIVE_ROOT.json').read_text(encoding='utf8'))['root'];src=root+'/seed%d'%seed;private=root.replace('experiment_workspaces','experiments')+'/seed%d/S'%seed
        local=local/('seed%d'%seed)
    case=dict(suite=suite,seed=seed,stage=stage,source_root=src,controller_root=root,private=private,arms=['C','S','Q','G']+(['U'] if suite=='OBJECT' else []),implementation={})
    dest=R/'reference_source'/name;dest.mkdir(parents=True,exist_ok=True)
    for n in ['runtime.py','layout.py','updates.py','memory.py','losses.py','legacy_losses.py','evaluate.py','train.py']:
        s=(local/n).read_text(encoding='utf8');(dest/n).write_text(s,encoding='utf8');case['implementation'][n]=hashlib.sha256(s.encode()).hexdigest()
    cases[name]=case
(R/'CASES.json').write_text(json.dumps(cases,indent=2)+'\n',encoding='utf8')
(R/'RUN_STATUS.json').write_text(json.dumps(dict(status='PARTIAL_PREPARATION',new_training_updates=0,new_environment_episodes=0),indent=2)+'\n')
print('Prepared local references and credential transport without remote dispatch.')
