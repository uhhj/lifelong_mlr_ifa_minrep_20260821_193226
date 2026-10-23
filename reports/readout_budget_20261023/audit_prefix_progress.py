"""Bind downloaded prefix metadata to accepted v2 scalar cases; no model forwards."""
import sys,json,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parent;S=Path(sys.argv[1]);V=R.parent/'20261023_readout_patch_v2'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dl=js(S/'DOWNLOAD.json');parent=V/'snapshot_20261023_160028';pd=js(parent/'DOWNLOAD.json')
def equal(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert math.isclose(a,b,abs_tol=1e-4,rel_tol=1e-5),(a,b)
 else:assert a==b
rows=[];case_hashes={}
for p in sorted((S/'metadata/cases').glob('*_b50.json')):
 name='cases/'+p.name;assert sha(p)==dl[name]['sha256'];oldp=parent/'metadata'/name;assert sha(oldp)==pd[name]['sha256']
 new,old=js(p),js(oldp);assert new['budget']==old['budget']==50 and new['kind']==old['kind'] and new['task']==old['task'];assert new['fold']==old['fold']
 assert new['selection']['selected_epoch']==old['selection']['selected_epoch'];equal(new['selection']['validation_score_m'],old['selection']['validation_score_m'])
 parity=new['v2_50_epoch_parity'];assert set(parity)=={'train','validation','test'};maxdiff=0.
 for part in parity:
  assert set(parity[part])==({'normal','latest_visual_previous','other_demo_visual'} if part=='test' else {'normal'})
  for mode,e in parity[part].items():
   assert e['allclose'] is True and e['parent_predictions_sha256']==old['predictions_sha256'] and math.isfinite(e['max_abs_prediction_m'])
   maxdiff=max(maxdiff,e['max_abs_prediction_m']);equal(new['partitions'][part][mode],old['partitions'][part][mode])
 rows.append(dict(case=p.name,selected_epoch=new['selection']['selected_epoch'],server_reported_max_abs_prediction_m=maxdiff,parent_prediction_sha256=old['predictions_sha256'],parent_case_sha256=sha(oldp)))
 case_hashes[name]=sha(p)
out=dict(status='PASS_LOCAL_PREFIX_SCALAR_AND_METADATA_BINDING_NOT_FINAL_CP_AUDIT',completed_prefixes=len(rows),rows=rows,case_sha256=case_hashes,snapshot_sha256=sha(S/'EVIDENCE.json'),download_sha256=sha(S/'DOWNLOAD.json'),parent_download_sha256=sha(parent/'DOWNLOAD.json'),model_forwards=0,optimizer_updates=0,environment_steps=0)
(S/'PREFIX_PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(dict(status=out['status'],completed_prefixes=len(rows),maximum_server_reported_difference_m=max([r['server_reported_max_abs_prediction_m'] for r in rows],default=None)),indent=2))
