"""Fresh asset binding for only S GOAL42/43. No writes, model forwards or environments."""
import os,sys,json,hashlib,time,subprocess,contextlib,io
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1')
P=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_FULL_GOAL41_20261025_v1_engineering02')
sys.path.insert(0,str(P))
from runtime import verify,torch,np,EasyDict
from libero.libero.benchmark import get_benchmark
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
assert verify()=='ef4f4b759632c7c81ebc5ca8da87011e1d4bad4fea48a205b9b30b7c2da1a54a'
assert (P/'controller.exit').read_text()=='0' and not Q.exists()
completed=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')
assert (completed/'engineering_copy_20261029_v1/controller.exit').read_text().strip()=='0'
parents={}
for seed in (41,42,43):
 out=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed%d_20261022'%seed)
 src=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_%d/source'%seed)
 pc=js(out/'CONTRACT.json');assert all(sha(src/n)==h for n,h in pc['source_sha256'].items())
 cfg=js(out/'resolved_ra_coreset.json');ec=js(out/'resolved_eval.json')
 assert int(cfg['seed'])==seed and int(ec['seed'])==seed
 assert ec['benchmark_name'].lower()=='libero_goal'
 with contextlib.redirect_stdout(io.StringIO()):bench=get_benchmark(ec['benchmark_name'])(ec['data']['task_order_index'])
 splits={};stages={};files={str(out/'CONTRACT.json'):sha(out/'CONTRACT.json')}
 for n in ['resolved_ra_coreset.json','resolved_eval.json','ra_coreset/INITIAL_STATE.json']:
  files[str(out/n)]=sha(out/n)
 for t in range(6,10):
  mf=out/('ra_coreset/dataset_manifests/task%d.json'%t);rows=js(mf)['rows'];files[str(mf)]=sha(mf)
  labels={}
  for prev in range(6,t):
   lp=out/('effects/memory/task%d.pth'%prev);meta=lp.with_suffix('.json');assert sha(lp)==js(meta)['sha256']
   files[str(lp)]=sha(lp);files[str(meta)]=sha(meta)
   labels.update({str(out/'ra_coreset/buffer'/n):v for n,v in torch.load(lp,map_location='cpu').items()})
   sf=out/('ra_coreset/storage/task%d.json'%prev);files[str(sf)]=sha(sf)
  lp=out/('effects/current/task%d/labels.pth'%t);meta=lp.parent/'AUDIT.json';assert sha(lp)==js(meta)['labels_sha256']
  files[str(lp)]=sha(lp);files[str(meta)]=sha(meta)
  labels.update({str(out/'features'/str(t)/n):v for n,v in torch.load(lp,map_location='cpu').items()})
  for row in rows:
   lab=labels[row['path']];assert sha(row['path'])==row['sha256']==lab['sample_sha256']
   assert all(row[k]==lab[k] for k in ['task','demo','u','bc_valid'])
   files[row['path']]=row['sha256']
  f=out/('ra_coreset/ifa_task%d.json'%t);files[str(f)]=sha(f)
  stages[t]=dict(population=len(rows),valid=sum(bool(x['bc_valid']) for x in rows),old_valid=sum(bool(x['bc_valid']) and x['task']<t for x in rows),batches=(len(rows)+9)//10,manifest_sha256=sha(mf))
  task=bench.get_task(t);ip=Path(ec['init_states_folder'])/task.problem_folder/task.init_states_file;arr=torch.load(ip,map_location='cpu')
  hs={str(i):hashlib.sha256(np.ascontiguousarray(np.asarray(arr[i])).tobytes()).hexdigest() for i in range(30)}
  training={str(i):hs[str(i)] for i in range(20,30)};evaluation={str(i):hs[str(i)] for i in range(20)}
  assert len(set(training.values()))==10 and not set(training.values())&set(evaluation.values())
  splits[t]=dict(path=str(ip),file_sha256=sha(ip),available=len(arr),training=training,evaluation=evaluation)
  files[str(ip)]=sha(ip);bd=Path(ec['bddl_folder'])/task.problem_folder/task.bddl_file;files[str(bd)]=sha(bd)
 for k in ['pretrain']:files[cfg[k]]=sha(cfg[k])
 assert sum(x['batches']*100 for x in stages.values())==76100
 parents[seed]=dict(output=str(out),source=str(src),parent_contract_sha256=sha(out/'CONTRACT.json'),source_sha256=pc['source_sha256'],files=files,stages=stages,initial_split=splits,official_initialization=dict(path=cfg['pretrain'],sha256=sha(cfg['pretrain'])))
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and (str(Q) in cmd or str(P) in cmd or str(completed) in cmd or 'SUPPORT_GATE_FORKS_20261027_v1' in cmd):processes.append(dict(pid=int(p.name),command=cmd))
 except OSError:pass
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cap={}
for mount,bound,reserve,rootfree in [('/root',0,1024**3,False),('/data',8*1024**3,2*1024**3,True)]:
 v=os.statvfs(mount);free=(v.f_bfree if rootfree else v.f_bavail)*v.f_frsize
 cap[mount]=dict(free=free,new_peak_bound=bound,reserve=reserve,headroom=free-bound-reserve)
print(json.dumps(dict(status='PASS_PREDEPLOY_READONLY' if not gpu and not processes and all(v['headroom']>=0 for v in cap.values()) else 'REVIEW_RESOURCE_REQUIRED',observed=time.time(),seed41_contract=verify(),parents=parents,capacity=cap,gpu=gpu,processes=processes,hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True),remote_writes=0,forwards=0,updates=0,environments=0)))
