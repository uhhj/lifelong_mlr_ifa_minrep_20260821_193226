"""Prepare finite C-only suite/seed extension; no remote dispatch."""
from pathlib import Path
import ast, json
R=Path(__file__).resolve().parent
W=R.parent
S=W/'20261026_ccbp_v4_s_object41_42_43'
NAME='CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1'
Q='/data/mlr_ifa/experiment_workspaces/'+NAME
CASES=['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']
def put(n,s):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
for n in ['remote.py','credential_session.py','credential_store.py','queue_remote.py','publish.py']:
 s=(S/n).read_text(encoding='utf8')
 if n=='publish.py':s=s.replace('reports/ccbp_v4_s_object41_42_43_20261026','reports/c_goal42_43_object41_42_43_20261027').replace('CCBP-v4 S OBJECT41-43 fixed-protocol extension','C GOAL42-43 OBJECT41-43 fixed-protocol extension')
 put(n,s)
s=(S/'preflight_readonly.py').read_text(encoding='utf8')
s=s.replace('/data/mlr_ifa/experiment_workspaces/CCBP_V4_S_OBJECT41_42_43_20261026_v1',Q)
s=s.replace('for seed in (41,42,43):',"for case in "+repr(CASES)+":\n suite,seed=case.split('_');seed=int(seed)")
a=s.index(" out=Path(");b=s.index(" pc=js(",a)
s=s[:a]+" out=Path('/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01' if case=='OBJECT_41' else '/data/mlr_ifa/experiments/ALIGN_%s_seed%d_20261022'%(suite,seed))\n src=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source' if case=='OBJECT_41' else '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/%s/source'%case)\n"+s[b:]
s=s.replace("=='LIBERO_OBJECT'","=='LIBERO_'+suite").replace('==105500',"==(76100 if suite=='GOAL' else 105500)").replace('parents[seed]=dict(',"parents[case]=dict(suite=suite,seed=seed,")
# Include exact ordered manifest metadata for independent local audits; no private payload.
s=s.replace('manifest_sha256=sha(mf))','manifest_sha256=sha(mf),rows=rows)')
# Only formal initial states are used by C; no training interaction.
s=s.replace('range(30)','range(20)').replace("training={str(i):hs[str(i)] for i in range(20,30)};evaluation={str(i):hs[str(i)] for i in range(20)}","training={};evaluation={str(i):hs[str(i)] for i in range(20)}").replace('assert len(set(training.values()))==10 and not set(training.values())&set(evaluation.values())','assert len(evaluation)==20')
# Verify actual accepted source implementation and completed latest experiment as prerequisites.
mark='parents={}';add="""C=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1_engineering02')
assert sha(C/'CONTRACT.json')=='232a34dbef9b268f4be82580f0b4299a2ecb944f436c1aac8fc146088f41243b'
assert all(sha(p)==h for p,h in js(C/'CONTRACT.json')['files'].items())
assert (C/'controller.exit').read_text()=='0'
LAST=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_S_OBJECT41_42_43_20261026_v1')
assert (LAST/'controller.exit').read_text()=='0'
implementation={n:sha(C/n) for n in ['layout.py','runtime.py','legacy_losses.py','losses.py','directional_prototype.py','updates.py','train.py','evaluate.py','synthetic_checks.py']}
parents={}"""
s=s.replace(mark,add).replace('str(P) in cmd','str(P) in cmd or str(C) in cmd or str(LAST) in cmd')
s=s.replace('parents=parents,capacity=cap','parents=parents,c_implementation=implementation,c_contract=sha(C/\'CONTRACT.json\'),capacity=cap')
ast.parse(s,feature_version=(3,8));put('preflight_readonly.py',s)
put('RUN_STATUS.json',json.dumps(dict(status='PREPARING_NOT_DISPATCHED',cases=CASES,training_epochs=2000,main_updates=468700,formal_episodes=1000,training_collection=0),indent=2)+'\n')
print('Prepared local transport and read-only preflight; no scientific execution.')
