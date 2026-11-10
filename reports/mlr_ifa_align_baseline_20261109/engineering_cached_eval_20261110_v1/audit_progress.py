from pathlib import Path
import json,sys,subprocess,hashlib
R=Path(__file__).resolve().parent;B=R.parent;D=Path(sys.argv[1]);J=lambda p:json.loads(p.read_text(encoding='utf8'))
subprocess.run([sys.executable,str(B/'engineering_assets_resume_20261110_v2/audit_progress.py'),*sys.argv[1:]],check=True)
e=J(D/'EVIDENCE.json');L={n:v['text'] for n,v in e['light'].items()};c=json.loads(L['cached_engineering/CONTRACT.json'])
assert e['cache_contract_sha256']==hashlib.sha256(L['cached_engineering/CONTRACT.json'].encode()).hexdigest()
assert 'cached_engineering/FAILURE.json' not in L and 'cached_engineering/WATCHDOG_RELEASE.json' not in L
assert all(v.strip()=='0' for n,v in L.items() if n.startswith('cached_engineering/') and n.endswith('.exit'))
events=[]
for prefix in ['engineering/','cached_engineering/']:events.extend(json.loads(x) for x in L.get(prefix+'EVENTS.jsonl','').splitlines())
events.sort(key=lambda x:x['time']);held=None;done=0
for x in events:
    if x['kind']=='environment_acquired':assert held is None;held=(x['case'],x['pid'])
    if x['kind']=='environment_released':assert held==(x['case'],x['pid']) and x['code']==0;held=None;done+=1
admissions={}
for n,v in L.items():
    if n.startswith('cached_engineering/admissions/') and n.endswith('.json'):
        a=json.loads(v);assert a['status']=='PASS_ACCEPTED_S_CACHE_TOLERANCE' and a['atol']==a['rtol']==1e-4 and a['model_unchanged'];assert a['optimizer_calls']==a['environments']==0
        admissions.setdefault(a['case'],[]).append(a)
handoff=json.loads(L['cached_engineering/HANDOFF_COMPLETE.json']) if 'cached_engineering/HANDOFF_COMPLETE.json' in L else None
if handoff:
    assert handoff['root_lock'] and handoff['returncode']==0 and handoff['evaluation_signals']==0
    assert 'engineering/controller.exit' not in L
out=dict(status='PASS_CACHED_ENGINEERING_PROGRESS_NOT_FINAL',handoff=handoff,combined_environment_mutex=dict(completed=done,active=held),cache_admissions={k:len(v) for k,v in admissions.items()},cached_cases=[k for k in admissions],science_contract_unchanged=True,no_completed_rollouts_repeated=True,evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest())
(D/'CACHED_ENGINEERING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
