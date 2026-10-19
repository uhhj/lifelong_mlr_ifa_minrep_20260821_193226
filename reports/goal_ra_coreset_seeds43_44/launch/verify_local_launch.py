"""Read-only consistency of reviewed code, downloaded launch proof and package hashes."""
from pathlib import Path
import json,hashlib,datetime,sys
R=Path(__file__).resolve().parent;B=R.parent/'20261018_ra_coreset_goal_seed42'
if not B.exists():B=R.parents[1]/'goal_ra_coreset_seed42/launch'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=js(R/'SOURCE_DIFF.json')
for n,h in proof['unchanged'].items():assert sha(R/'source'/n)==sha(B/'source'/n)==h
for n,h in proof['changed'].items():assert sha(R/'source'/n)==h
assert set(proof['changed'])=={'ra_exp/prepare.py','ra_exp/select_offline.py','ra_exp/freeze.py','ra_exp/audit.py'}
for n,h in js(R/'CODE_MANIFEST.json').items():
 parts=Path(n).parts
 p=R/'source'/Path(*parts[2:]) if len(parts)>2 and parts[1]=='source' else R/n
 assert sha(p)==h,n
snapshot=js(R/'CURRENT_SNAPSHOT.json');checked={}
for seed in (43,44):
 dirs=sorted(R.glob('progress_*/seed%d/evidence'%seed))
 if not dirs:
  checked[str(seed)]=dict(status='PENDING_ADMISSION_OR_COLLECTION');continue
 E=dirs[-1];manifest=js(E/'PACKAGE_MANIFEST.json')
 for n,m in manifest['files'].items():assert sha(E/n)==m['sha256'] and (E/n).stat().st_size==m['bytes'],n
 c=js(E/'CONTRACT.json');assert c['seed']==seed
 for n,h in c['source_sha256'].items():assert sha(R/'source'/n)==h,n
 pre=js(E/'GPU_PREFLIGHT.json');assert pre['status']=='PASS' and pre['isolated_optimizer_calls']==4 and pre['environment_steps']==0 and pre['contract_sha256']==sha(E/'CONTRACT.json')
 off=js(E/'OFFLINE_ACCEPTANCE.json');assert off['status']=='PASS' and sum(x['N'] for x in off['tasks'].values())==4759
 for key in ['resolved_eval.json','resolved_ra_coreset.json']:
  conf=js(E/key);assert conf['seed']==seed and conf['gamma']==.7
  assert all(conf['policy'][k] is True for k in ['use_vision','use_language','use_state'])
 for t,v in off['tasks'].items():
  assert v['independent_all_removal_choices'] and v['random_parent_order'] and v['coreset_parent_order']
  assert v['q']==[499,445,452,985][int(t)-6] and v['anchors']==v['q']//2
 checked[str(seed)]=dict(status='ADMISSION_PASS',package_files=len(manifest['files']),swaps={t:v['swaps'] for t,v in off['tasks'].items()},contract_sha256=sha(E/'CONTRACT.json'))
out=dict(status='PARTIAL_LAUNCH_EVIDENCE_PASS',time=datetime.datetime.now().astimezone().isoformat(),source_unchanged_files=len(proof['unchanged']),changed_plumbing_files=list(proof['changed']),seeds=checked,observation_time=snapshot['time'],no_new_rollouts=True)
if '--check-only' not in sys.argv:(R/'LOCAL_LAUNCH_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
