"""Read-only state, exit, process, disk, GPU and count snapshot."""
import json,time,subprocess,shutil
from pathlib import Path
W=Path('/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018');O=Path('/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018')
def js(p):return json.loads(p.read_text()) if p.exists() else None
def main():
    epochs=js(O/'epochs.json') or [];trials=[];attempts=[]
    for p in (O/'evaluation').glob('trials*.jsonl'):trials.extend(json.loads(s) for s in p.read_text().splitlines())
    for p in (O/'evaluation').glob('attempts*.jsonl'):attempts.extend(json.loads(s) for s in p.read_text().splitlines())
    proc=subprocess.check_output(['ps','-eo','pid,etime,pcpu,pmem,args'],text=True)
    lines=[l for l in proc.splitlines() if ('python' in l and ('-m zr2_exp.' in l or str(W/'pipeline.py') in l)) and 'ps -eo' not in l]
    x=dict(time=time.time(),status=js(O/'RUN_STATUS.json'),epochs=len(epochs),latest_epoch=epochs[-1] if epochs else None,optimizer_steps=sum(r['batches'] for r in epochs),formal_records=len(trials),completed_vectors=sum(r['event']=='vector_attempt_complete' for r in attempts),technical_exceptions=sum(r['event']=='technical_exception' for r in attempts),exit_codes={n:(O/(n+'.exit')).read_text().strip() if (O/(n+'.exit')).exists() else None for n in ['pipeline','calibrate','evaluate','audit']},processes=lines,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True),free_bytes={p:shutil.disk_usage(p).free for p in ['/data','/root']},structure_checks=len((O/'STRUCTURE.jsonl').read_text().splitlines()) if (O/'STRUCTURE.jsonl').exists() else 0)
    print(json.dumps(x,indent=2))
if __name__=='__main__':main()
