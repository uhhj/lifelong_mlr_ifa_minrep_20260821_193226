import sys,json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]).resolve()
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=js(D/'SNAPSHOT.json');ct=js(R/'CONTRACT.json');active=js(R/'ACTIVE_ROOT.json')
assert d['source_unchanged'] and d['contract_sha256']==active['contract_sha256']==sha(R/'CONTRACT.json')
for n,h in ct['files'].items():assert sha(R/n)==h
files=d['files'];events=[json.loads(x) for x in files.get('EVENTS.jsonl','').splitlines()];rec=json.loads(files.get('OUTCOMES.json','[]'))
assert len({(r['seed'],r['method'],r['driver'],r['initial_index']) for r in rec})==len(rec)
for r in rec:
 assert r['seed'] in [41,42,43] and r['method'] in ['RA','ALIGN'] and r['driver'] in [8,9] and 0<=r['initial_index']<20
 assert r['attempt']==1 and r['same_history_reconstruction'] and 1<=r['policy_steps']<=600
 if not r['success']:assert r['policy_steps']==600 and r['termination']=='timeout600'
ad=json.loads(files['ADMISSION.json']) if 'ADMISSION.json' in files else None
if ad:
 assert ad['status']=='PASS_OWN_GPU_ADMISSION_ZERO_ENV_UPDATE' and ad['updates']==ad['environments']==0
 assert len(ad['strict_checkpoints'])==12 and all(x['strict_keys']==100 and x['lengths']==list(range(1,9)) for x in ad['strict_checkpoints'])
 assert len(ad['cache_parity'])==8
done=[x for x in events if x['kind']=='vector_complete'];starts=[x for x in events if x['kind']=='vector_start']
assert len(done)*10==len(rec)
failed='FAILURE.json' in files or files.get('worker.exit','0').strip()!='0'
closed=files.get('worker.exit','').strip()=='0' and files.get('supervisor.exit','').strip()=='0' and 'COMPLETE.json' in files
result={'status':'ENGINEERING_FAILURE' if failed else 'COMPLETE_REMOTE_PENDING_FINAL_LOCAL' if closed else 'PASS_PROGRESS_NOT_FINAL','observation':datetime.datetime.fromtimestamp(d['time']).astimezone().isoformat(),'episodes':len(rec),'vectors':len(done),'admission':ad is not None,'active_vector':starts[-1]['run'] if starts and len(starts)>len(done) else None,'training_updates':0,'sources':{p.name:sha(p) for p in D.iterdir() if p.is_file()},'exit':files.get('worker.exit'),'gpu':d['gpu']}
if done:
 durations=[x['seconds'] for x in done];mean=sum(durations)/len(durations);result['mean_vector_seconds']=mean;result['remaining_linear_seconds']=(24-len(done))*mean
 if len(starts)>len(done):result['remaining_linear_seconds']=max(0,result['remaining_linear_seconds']-(d['time']-starts[-1]['time']))
(D/'LOCAL_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
status={'status':'PARTIAL_ENGINEERING_FAILURE' if failed else 'PARTIAL_FINAL_AUDIT_PENDING' if closed else 'PARTIAL_PAIRED_DIAGNOSTIC_RUNNING','observation':result['observation'],'offline':'COMPLETE_LOCAL_ACCEPTED','closed_loop':result,'active_root':active,'training_updates':0,'conditional_protection_training':'NOT_STARTED'}
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
