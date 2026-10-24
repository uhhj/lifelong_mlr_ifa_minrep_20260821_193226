import os,subprocess,json,time,hashlib,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_GOAL41_20261024_v1')
P=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1')
O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022')
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not Q.exists() and not P.exists(),'Version already exists; no redeploy'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert sha(O/'CONTRACT.json')=='f556347a93aeeb17bfad177fc3088535bd03ae259e7198a40a805ac721b8bdc8'
pc=json.loads((O/'CONTRACT.json').read_text());assert all(sha(S/n)==h for n,h in pc['source_sha256'].items())
assert json.loads((O/'PROCESS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
assert sha(O/'ra_coreset/models/mlrifa_8_model_ep99.pth')=='7c4e99ef40269d6d2169459d17818bb526afe288188372a7a6ebc2abf0247f3a'
a=os.statvfs('/root');b=os.statvfs('/data');assert a.f_bavail*a.f_frsize>3.3*1024**3 and b.f_bfree*b.f_frsize>2*1024**3
lock=open('/data/mlr_ifa/CCBP_GOAL41_20261024.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,v in BUNDLE.items():
 assert '/' not in n and Path(n).suffix in ['.py','.md','.txt']
 if n.endswith('.py'):compile(v,n,'exec')
Q.mkdir();P.mkdir();files={}
for n,v in BUNDLE.items():(Q/n).write_text(v);files[str(Q/n)]=sha(Q/n)
for n in ['CONTRACT.json','PROCESS_EXIT_ACCEPTANCE.json','resolved_ra_coreset.json','resolved_eval.json','ra_coreset/models/mlrifa_8_model_ep99.pth','ra_coreset/dataset_manifests/task9.json','ra_coreset/ifa_task9.json','effects/current/task9/labels.pth','effects/current/task9/AUDIT.json']+['effects/memory/task%d.%s'%(t,e) for t in [6,7,8] for e in ['pth','json']]:files[str(O/n)]=sha(O/n)
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,source_root=str(S),parent_source=pc['source_sha256'],created=time.time(),budget=dict(arms=list('ABCD'),seed=41,epochs=400,steps=134800,episodes=160)),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_NOT_YET_ADMITTED',root=str(Q),private=str(P),controller=p.pid,contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),observed=time.time())))
