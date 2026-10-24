import os,subprocess,json,time,hashlib,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1')
PR=Path('/root/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1');PD=Path('/data/mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1')
O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022');S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
OLD=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
assert not any(p.exists() for p in (Q,PR,PD)),'No redeployment'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert sha(OLD/'CONTRACT.json')=='fa6886806120e9cff7547fc0723b602e4fd53497c0a9c1d433618e50be267e47' and (OLD/'controller.exit').read_text()=='1'
oc=json.loads((OLD/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in oc['files'].items());assert all(sha(S/n)==h for n,h in oc['parent_source'].items())
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 assert str(OLD) not in cmd
assert sha(O/'CONTRACT.json')=='f556347a93aeeb17bfad177fc3088535bd03ae259e7198a40a805ac721b8bdc8'
assert json.loads((O/'PROCESS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
cleanup=Path('/data/mlr_ifa/experiment_workspaces/CCBP_FULL_REPLACEMENT_CLEANUP_20261024.json');assert json.loads(cleanup.read_text())['status']=='EXACT_OBSOLETE_RECOVERY_CLEANUP_COMPLETE'
cap={}
for mount,bound,reserve,rootfree in [('/root',int(1.07*1024**3),1024**3,False),('/data',int(1.88*1024**3),2*1024**3,True)]:
 st=os.statvfs(mount);free=(st.f_bfree if rootfree else st.f_bavail)*st.f_frsize;assert free-bound>=reserve,(mount,free,bound,reserve);cap[mount]=dict(free=free,new_peak_bound=bound,reserve=reserve,remaining=free-bound-reserve)
lock=open('/data/mlr_ifa/CCBP_V2_FULL_GOAL41_20261024.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,v in BUNDLE.items():
 assert '/' not in n and Path(n).suffix in ['.py','.md','.txt']
 if n.endswith('.py'):compile(v,n,'exec')
Q.mkdir();PR.mkdir();PD.mkdir();files={}
for n,v in BUNDLE.items():(Q/n).write_text(v);files[str(Q/n)]=sha(Q/n)
names=['CONTRACT.json','PROCESS_EXIT_ACCEPTANCE.json','resolved_ra_coreset.json','resolved_eval.json','ra_coreset/INITIAL_STATE.json','ra_coreset/models/mlrifa_6_model_ep99.pth']
for t in range(6,10):
 names+=['ra_coreset/dataset_manifests/task%d.json'%t,'ra_coreset/ifa_task%d.json'%t,'effects/current/task%d/labels.pth'%t,'effects/current/task%d/AUDIT.json'%t]
 if t<9:names+=['effects/memory/task%d.pth'%t,'effects/memory/task%d.json'%t,'ra_coreset/storage/task%d.json'%t]
for n in names:files[str(O/n)]=sha(O/n)
files[str(cleanup)]=sha(cleanup)
official=Path(json.loads((O/'resolved_ra_coreset.json').read_text())['pretrain'])
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,source_root=str(S),parent_source=oc['parent_source'],official_initialization=dict(path=str(official),sha256=sha(official)),created=time.time(),budget=dict(arms=list('BCNV'),seed=41,epochs=1600,steps=304400,episodes=800),capacity=cap),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_NOT_YET_ADMITTED',root=str(Q),private_root=str(PR),private_data=str(PD),controller=p.pid,contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),capacity=cap,observed=time.time())))
