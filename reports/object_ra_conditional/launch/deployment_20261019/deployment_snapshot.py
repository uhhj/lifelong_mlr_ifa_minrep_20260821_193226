import json,time,subprocess,sys,hashlib
from pathlib import Path
N=Path('/data/mlr_ifa/experiment_workspaces/RA_ZRv2_autonomous_20261019');sys.path.insert(0,str(N))
from flow_support import verify_controller,alive
verify_controller()
x={'status':'PARTIAL','time':time.time(),'frozen_controller_hashes':'PASS','processes':{},'seeds':{}}
for name in ['AUTONOMOUS_LAUNCH.json','wrapper.started','controller.started','recovery_launch.json']:
 z=json.loads((N/name).read_text());x['processes'][name]={'pid':z['pid'],'alive':alive(z['pid'])}
x['controller_status']=json.loads((N/'STATUS.json').read_text())
for s in [41,42,43]:
 O=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%s);e=O/'ra_coreset/epochs.json';rows=json.loads(e.read_text()) if e.exists() else []
 ev=O/'ra_coreset/evaluation';fs=list(ev.glob('trials*.jsonl'))
 x['seeds'][str(s)]={'epochs':len(rows),'logical_steps':sum(a['batches'] for a in rows),'last_epoch':rows[-1] if rows else None,'episode_files':len(fs),'records':sum(len(f.read_text().splitlines()) for f in fs),'external_exits':{p.name:p.read_text().strip() for p in (N/'stage1'/str(s)).glob('*.exit')}}
x['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
x['disk']=subprocess.check_output(['df','-h','/data','/'],text=True)
x['workers']=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines()
x['workers']=[a for a in x['workers'] if any(v in a for v in ['ra_adapter.py','ra_driver.py','autonomous_20261019/supervisor.py','autonomous_20261019/controller.py','coreset_exp.evaluate'])]
(N/'DEPLOYMENT_PROGRESS.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
