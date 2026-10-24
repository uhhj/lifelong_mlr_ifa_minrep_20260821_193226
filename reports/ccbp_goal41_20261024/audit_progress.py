import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];p=D/'light'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x['text'],encoding='utf8');light[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
failure=light.get('FAILURE.json');counts={};rates={}
if not failure and light.get('controller.exit')!='0' and not v['processes']:
 failure={'reason':'Controller absent without successful exit; inspect before any recovery'}
parent=R.parent/'20261022_align_object_extension_goal_conditional/final_GOAL_41_20261023_052225/current/ra_coreset/dataset_manifests/task9.json'
manifest=json.loads(parent.read_text(encoding='utf8'))['rows'];assert len(manifest)==3365
expected_valid=sum(r['bc_valid'] for r in manifest);expected_old=sum(r['bc_valid'] and r['task']==8 for r in manifest)
if 'TEACHER.json' in light:
 assert light['TEACHER.json']['manifest_sha256']==hashlib.sha256(parent.read_bytes()).hexdigest()
 assert light['TEACHER.json']['count']==expected_old
for arm in 'ABCD':
 rows=light.get('EPOCHS_'+arm+'.json',[])
 assert all(r['epoch']==i and r['steps']==(i+1)*337 and r['batches']==337 and r['totals']['valid']+r['totals']['invalid']==3365 for i,r in enumerate(rows))
 assert all(r['totals']['valid']==expected_valid and r['totals']['old_eligible']==expected_old for r in rows)
 counts[arm]=dict(epochs=len(rows),steps=len(rows)*337)
 if rows:rates[arm]=sum(x['seconds'] for x in rows[-10:])/len(rows[-10:])
for arm in 'BCD':
 for x,y in zip(light.get('EPOCHS_A.json',[]),light.get('EPOCHS_'+arm+'.json',[])):
  assert x['sampler_sha256']==y['sampler_sha256']
  assert all(x['totals'][k]==y['totals'][k] for k in ['valid','invalid','old_eligible'])
out=dict(status='FAILURE_REQUIRES_INSPECTION' if failure else 'PASS_PROGRESS_NOT_FINAL',observation=v['observed'],counts=counts,epoch_seconds_last10=rates,phase=light.get('STATUS.json'),failure=failure,processes=v['processes'],contract=v['contract_sha256'],independent_parent_manifest_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),expected_counts=dict(population=3365,valid=expected_valid,old_task8_valid=expected_old))
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
