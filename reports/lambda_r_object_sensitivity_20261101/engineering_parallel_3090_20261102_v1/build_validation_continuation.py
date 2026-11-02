"""Versioned continuation of missing parallel diagnostic only; no serial replay."""
from pathlib import Path
import ast,json,hashlib
E=Path(__file__).resolve().parent
s=(E/'full_driver.py').read_text(encoding='utf8')
s=s.replace("assert not (D/'STARTED.json').exists(),'single use';put(D/'STARTED.json',", "assert not (D/'CONTINUATION_STARTED.json').exists(),'single use';put(D/'CONTINUATION_STARTED.json',")
needle=" for seed in [42,43]:\n  guard_capacity();mode='serial'+str(seed);phase([launch(seed,mode)],mode)"
assert s.count(needle)==1
s=s.replace(needle," global events,metrics\n events=js(D/'EVENTS.json');metrics=js(D/'METRICS.json')\n assert not any(e['event']=='launch' and e['mode']=='parallel' for e in events)\n assert all(js(D/('serial'+str(seed)+'_'+str(seed))/'RESULT.json')['optimizer_completed']==3196 for seed in [42,43])")
old="results=[js(t/'RESULT.json') for p,t in children];calls=sum(x['optimizer_completed'] for x in results);assert calls==12784 and all(x['optimizer_started']==x['optimizer_completed'] for x in results)"
new="results=[js(t/'RESULT.json') for p,t in children];current=sum(x['optimizer_completed'] for x in results);assert current==6392 and all(x['optimizer_started']==x['optimizer_completed'] for x in results)\n retained=[js(D/('serial'+str(seed)+'_'+str(seed))/'RESULT.json') for seed in [42,43]];assert all(x['optimizer_started']==x['optimizer_completed']==3196 for x in retained);calls=current+sum(x['optimizer_completed'] for x in retained)"
assert old in s;s=s.replace(old,new)
s=s.replace("ended=time.time(),reports=reports,admission=", "ended=time.time(),reports=reports,continuation_current_optimizer_calls=current,retained_serial_optimizer_calls=6392,resolved_guard_failure_sha256=hashlib.sha256((D/'FAILURE.json').read_bytes()).hexdigest(),continuation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),admission=")
s=s.replace("put(D/'FAILURE.json',", "put(D/'CONTINUATION_FAILURE.json',")
ast.parse(s);(E/'full_continue_v2.py').write_text(s,encoding='utf8')
dispatch='''"""Unique diagnostic continuation, completed serial references remain untouched."""
import json,sys,hashlib,subprocess,os,time
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_parallel_full_20261102_v1')
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
SOURCE=__SOURCE__
js=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (D/'CONTINUATION_DISPATCH.json').exists() and not (D/'continue_parallel_v2.py').exists() and not (D/'RESULT.json').exists()
f=js(D/'FAILURE.json');assert 'Insufficient diagnostic headroom' in f['error'] and 'guard_capacity()' in f['error']
assert len(f['diagnostic_children'])==2 and all(x['returncode']==0 for x in f['diagnostic_children'])
manifest=js(D/'SOURCE_MANIFEST.json');assert all(sha(Path(p))==h for p,h in manifest.items())
assert sha(D/'worker.py')=='8aea9661f4f690a545e121a4605f79a61b05bef908513f41da56c2039a092f23' and sha(D/'driver.py')=='ad37ea3b4797b6a325f15b2474d5892fb6ca10da294c71b58f55bac041fb6bb9'
refs={}
for seed in [42,43]:
 p=D/('serial'+str(seed)+'_'+str(seed))/'RESULT.json';r=js(p);assert r['optimizer_started']==r['optimizer_completed']==3196 and r['environments']==r['extra_autograd']==0
 assert len(r['stages'])==7 and all(sha(Path(n))==h for n,h in r['baseline_hashes'].items())
 refs[str(p)]=sha(p)
 assert not (D/('parallel_'+str(seed))).exists()
for item in [js(D/'DISPATCH.json'),*f['diagnostic_children']]:
 p=Path('/proc')/str(item['pid']);assert not p.exists() or (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z'
for name in ['CONTRACT.json','engineering_device_snapshot_20261101_v1/CONTRACT.json','engineering_resume_3090_20261102_v1/CONTRACT.json','engineering_checks_3090_20261102_v1/CONTRACT.json']:assert all(sha(Path(p))==h for p,h in js(Q/name)['files'].items())
mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1]);g=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert g.returncode==0 and mem>=12*1024**2 and float(g.stdout.strip())>=10000
status=js(Q/'L005_42/TRAIN_STATUS.json');assert status['phase']=='train' and status['pid']==4694
(D/'continue_parallel_v2.py').write_text(SOURCE)
with (D/'continue_parallel_v2.log').open('x') as log:p=subprocess.Popen([sys.executable,'-B',str(D/'continue_parallel_v2.py')],cwd=D,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
r=dict(status='CONTINUE_MISSING_PARALLEL_ONLY',pid=p.pid,time=time.time(),source_sha256=sha(D/'continue_parallel_v2.py'),retained_serial_result_hashes=refs,retained_optimizer_calls=6392,remaining_optimizer_calls=6392,full_optimizer_calls=12784,prior_guard_failure_sha256=sha(D/'FAILURE.json'),original_driver_sha256=sha(D/'driver.py'),production_signals=0,environments=0)
(D/'CONTINUATION_DISPATCH.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''.replace('__SOURCE__',repr(s))
ast.parse(dispatch);(E/'dispatch_full_continue_v2.py').write_text(dispatch,encoding='utf8')
status=(E/'full_status.py').read_text(encoding='utf8')
status=status.replace('source_hashes=source_hashes,driver=',"source_hashes=source_hashes,continuation_driver=proc(files['CONTINUATION_DISPATCH.json']['pid']),continuation_source_sha256=hashlib.sha256((D/'continue_parallel_v2.py').read_bytes()).hexdigest(),driver=")
ast.parse(status);(E/'full_status_v2.py').write_text(status,encoding='utf8')
print(json.dumps(dict(status='STATIC_CONTINUATION_REVIEW',source_sha256=hashlib.sha256(s.encode()).hexdigest(),retained=6392,remaining=6392,total=12784,no_completed_diagnostic_replay=True,original_failure_preserved=True)))
