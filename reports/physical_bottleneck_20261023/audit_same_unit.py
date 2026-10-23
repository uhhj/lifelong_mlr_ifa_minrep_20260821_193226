import json,hashlib,statistics as st
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'same_unit_readonly_20261023_150704';M=R.parent/'20261023_physical_effect_probe/snapshot_20261023_141307/metadata'
ev=json.loads((D/'EVIDENCE.json').read_text());orig=json.loads((M/'STATE_PROBE_RESULTS.json').read_text());assert ev['status']=='PASS_SAME_UNIT_EXISTING_CHECKPOINTS';assert ev['source_results_sha256']==hashlib.sha256((M/'STATE_PROBE_RESULTS.json').read_bytes()).hexdigest()
rows=[]
for key,md in ev['case_files'].items():
 p=D/(key+'.json');assert p.stat().st_size==md['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==md['sha256'];v=json.loads(p.read_text());rows.append(v)
 old=next(x for x in orig['results'] if x['model']==v['model'] and x['task']==v['task'] and x['train_view']==x['test_view']=='native');sel=old['folds'][v['fold']['fold']];assert sel['checkpoint_sha256']==v['checkpoint_sha256'] and sel['selected_epoch']==v['selected_epoch']
 for part in ('train','validation','test'):
  z=v['partitions'][part]['metrics'];assert z['all']['n']==sel[part+'_windows'];assert set(z['by_demo'])==set(v['fold'][part]);assert z['all']['n']==z['dynamic']['n']+z['static']['n']
 for demo,z in v['partitions']['test']['metrics']['by_demo'].items():
  for j,field in enumerate(('object_eef','object_goal','aperture')):
   expected=old['interventions']['normal'][field]['all']['by_demo'][demo];assert z['n']==expected['n'] and abs(z['median_mm'][j]-expected['median_mm'])<.02
assert len(rows)==50
summary=[]
for kind in ('P0','P1','P2','P3','P4'):
 for task in (6,7):
  rr=[x for x in rows if x['model']==kind and x['task']==task];assert len(rr)==5
  summary.append(dict(model=kind,task=task,partitions={part:{stat:[st.mean(v['partitions'][part]['metrics']['all'][stat][j] for v in rr) for j in range(3)] for stat in ('median_mm','p90_mm','mean_mm')} for part in ('train','validation','test')}))
out=dict(status='PASS_LOCAL_IDENTITIES_COUNTS_AND_TEST_REPLICATION',cases=50,fields=['object_eef','object_goal','aperture'],summary=summary,aggregation='Equal mean of per-fold metrics, not pooled quantile; train folds overlap',forward_batches=ev['model_forward_batches'],optimizer_steps=0,environment_episodes=0,evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),limitations='New train metrics are model-forward observations, not independently reexecuted by this local scalar audit.')
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(out['status'])
