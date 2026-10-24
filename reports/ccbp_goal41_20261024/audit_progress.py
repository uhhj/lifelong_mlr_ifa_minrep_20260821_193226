import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256'];p=D/'light'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x['text'],encoding='utf8');light[n]=json.loads(x['text']) if n.endswith('.json') else x['text']
failure=light.get('FAILURE.json');counts={};rates={}
for arm in 'ABCD':
 rows=light.get('EPOCHS_'+arm+'.json',[])
 assert all(r['epoch']==i and r['steps']==(i+1)*337 and r['batches']==337 and r['totals']['valid']+r['totals']['invalid']==3365 for i,r in enumerate(rows))
 counts[arm]=dict(epochs=len(rows),steps=len(rows)*337)
 if rows:rates[arm]=sum(x['seconds'] for x in rows[-10:])/len(rows[-10:])
for arm in 'BCD':
 for x,y in zip(light.get('EPOCHS_A.json',[]),light.get('EPOCHS_'+arm+'.json',[])):
  assert x['sampler_sha256']==y['sampler_sha256']
  assert all(x['totals'][k]==y['totals'][k] for k in ['valid','invalid','old_eligible'])
out=dict(status='FAILURE_REQUIRES_INSPECTION' if failure else 'PASS_PROGRESS_NOT_FINAL',observation=v['observed'],counts=counts,epoch_seconds_last10=rates,phase=light.get('STATUS.json'),failure=failure,processes=v['processes'],contract=v['contract_sha256'])
(D/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
