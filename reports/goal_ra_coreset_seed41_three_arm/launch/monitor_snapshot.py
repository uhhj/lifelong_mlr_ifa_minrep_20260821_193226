"""Read-only state inspection, no scientific computation or retries."""
from pathlib import Path
import json,subprocess,shutil,time
Q=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text()) if p.exists() else None
def txt(p):return p.read_text().strip() if p.exists() else None
out=dict(time=time.time(),queue_exit=txt(Q/'queue.exit'),wrapper_exit=txt(Q/'wrapper.exit'),failure=js(Q/'QUEUE_FAILURE.json'),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True),free_bytes={p:shutil.disk_usage(p).free for p in ['/root','/data']},arms={},phase_exits={p.name:txt(p) for p in Q.glob('*.exit')})
for arm,name in [('random','Coreset_paper'),('coreset','Coreset_paper'),('ra_coreset','RA_Coreset')]:
 O=Path('/root/mlr_ifa/experiments/'+name+'_GOAL_seed41_20261019');e=js(O/arm/'epochs.json') or [];ev=O/arm/'evaluation'
 out['arms'][arm]=dict(status=js(O/'RUN_STATUS.json'),epochs=len(e),latest=e[-1] if e else None,records=sum(len(p.read_text().splitlines()) for p in ev.glob('trials*.jsonl')),completed_vectors=sum(json.loads(l)['event']=='vector_attempt_complete' for p in ev.glob('attempts*.jsonl') for l in p.read_text().splitlines()),exits={n:txt(O/(n+'.exit')) for n in ['pipeline','supervisor']},preflight=js(O/'GPU_PREFLIGHT.json'))
out['logs']={str(p.relative_to(Q)):dict(size=p.stat().st_size,mtime=p.stat().st_mtime,tail=p.read_text(errors='replace').splitlines()[-5:]) for p in Q.glob('*.log')}
out['processes']=[x for x in subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines() if (str(Q) in x or ' -m coreset_exp.' in x) and 'ps -eo' not in x]
out['final_acceptance']=js(Q/'FINAL_QUEUE_ACCEPTANCE.json')
print(json.dumps(out,indent=2))
