"""Read-only compact current snapshot; no restart or model load."""
from pathlib import Path
import subprocess,json,shutil,time
Q=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text()) if p.exists() else None
def txt(p):return p.read_text().strip() if p.exists() else None
out=dict(time=time.time(),queue_exit=txt(Q/'queue.exit'),wrapper_exit=txt(Q/'wrapper.exit'),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True),free_bytes={p:shutil.disk_usage(p).free for p in ['/root','/data']},seeds={},queue_failure=js(Q/'QUEUE_FAILURE.json'))
for s in (43,44):
 O=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_20261019'%s);ep=js(O/'ra_coreset/epochs.json') or []
 trials=list((O/'ra_coreset/evaluation').glob('trials*.jsonl'));events=[json.loads(x) for p in (O/'ra_coreset/evaluation').glob('attempts*.jsonl') for x in p.read_text().splitlines()]
 out['seeds'][s]=dict(status=js(O/'RUN_STATUS.json'),epochs=len(ep),latest=ep[-1] if ep else None,records=sum(len(p.read_text().splitlines()) for p in trials),completed_vectors=sum(x['event']=='vector_attempt_complete' for x in events),technical_exceptions=sum(x['event']=='technical_exception' for x in events),exits={n:txt(O/(n+'.exit')) for n in ['train','evaluate','audit','pipeline','supervisor']},admission_exit=txt(Q/('seed%d_admission.exit'%s)),gpu_preflight=js(O/'GPU_PREFLIGHT.json'))
out['processes']=[x for x in subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines() if (' -m coreset_exp.' in x or str(Q) in x) and 'ps -eo' not in x]
print(json.dumps(out,indent=2))
