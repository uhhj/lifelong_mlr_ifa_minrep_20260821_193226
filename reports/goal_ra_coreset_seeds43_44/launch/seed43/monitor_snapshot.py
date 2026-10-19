from pathlib import Path
import os,json,time,subprocess,shutil
O=Path(os.environ['CORESET_OUT'])
def js(p):return json.loads(p.read_text()) if p.exists() else None
def main():
    epochs=js(O/'ra_coreset/epochs.json') or [];trials=list((O/'ra_coreset/evaluation').glob('trials*.jsonl'));attempts=[json.loads(x) for p in (O/'ra_coreset/evaluation').glob('attempts*.jsonl') for x in p.read_text().splitlines()]
    processes=[x for x in subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines() if (' -m coreset_exp.train' in x or ' -m coreset_exp.evaluate' in x or str(O).split('/')[-1] in x and 'pipeline.py' in x) and 'ps -eo' not in x]
    print(json.dumps(dict(time=time.time(),status=js(O/'RUN_STATUS.json'),epochs=len(epochs),latest_epoch=epochs[-1] if epochs else None,records=sum(len(p.read_text().splitlines()) for p in trials),completed_vectors=sum(r['event']=='vector_attempt_complete' for r in attempts),technical_exceptions=sum(r['event']=='technical_exception' for r in attempts),exits={n:(O/(n+'.exit')).read_text().strip() if (O/(n+'.exit')).exists() else None for n in ['train','evaluate','audit','pipeline']},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True),processes=processes,free_bytes={x:shutil.disk_usage(x).free for x in ['/root','/data']}),indent=2))
if __name__=='__main__':main()
