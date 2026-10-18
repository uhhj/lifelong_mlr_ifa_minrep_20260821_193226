"""External read-only queue exit, complete-count and artifact acceptance. No models run."""
import json,time,subprocess,hashlib
from pathlib import Path
Q=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (Q/'queue.exit').read_text().strip()=='0'
pid=int((Q/'queue.pid').read_text())
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
assert not any(x.strip().split(maxsplit=1)[0]==str(pid) for x in ps)
assert not any(' -m coreset_exp.'+n in x for x in ps for n in ('train','export','evaluate','preflight'))
for p,h in json.loads((Q/'ORCHESTRATION_SHA256.json').read_text()).items():assert sha(Path(p))==h,p
all_counts={};inputs={}
for s in (42,43,44):
    O=Path(f'/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{s}_20261017')
    for n in ('pipeline.exit','supervisor.exit'):assert (O/n).read_text().strip()=='0'
    a=json.loads((O/'INDEPENDENT_ACCEPTANCE.json').read_text());assert a['status']=='PASS' and a['final_checkpoints']==8
    summary=json.loads((O/'SUMMARY.json').read_text());assert summary['formal_records']==400
    counts={}
    for arm in ('random','coreset'):
        epochs=json.loads((O/arm/'epochs.json').read_text());assert len(epochs)==400 and epochs[-1]['steps']==76100
        records=[];attempts=[]
        for p in sorted((O/arm/'evaluation').glob('trials*.jsonl')):
            records.extend(json.loads(x) for x in p.read_text().splitlines());inputs[str(p)]=sha(p)
        for p in sorted((O/arm/'evaluation').glob('attempts*.jsonl')):
            attempts.extend(json.loads(x) for x in p.read_text().splitlines());inputs[str(p)]=sha(p)
        assert len(records)==200 and sum(x['event']=='vector_attempt_complete' for x in attempts)==20
        assert not any(x['event']=='technical_exception' for x in attempts)
        counts[arm]=dict(epochs=400,steps=76100,checkpoints=4,episodes=200,vector_attempts=20)
    all_counts[str(s)]=counts
aggregate=json.loads((Q/'THREE_SEED_SUMMARY.json').read_text());assert aggregate['formal_records']==1200 and aggregate['optimizer_steps']==456600
for p,h in aggregate['inputs'].items():assert sha(Path(p))==h
out=dict(status='PASS',queue_exit=0,seeds=all_counts,total_epochs=2400,total_optimizer_steps=456600,
         final_checkpoints=24,formal_records=1200,vector_attempts=120,inputs=inputs,unix=time.time())
(Q/'FINAL_QUEUE_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
(Q/'RUN_STATUS.json').write_text(json.dumps(dict(status='COMPLETED',unix=time.time(),formal_records=1200,acceptance='FINAL_QUEUE_ACCEPTANCE.json'),indent=2)+'\n')
print('PASS: 2400 epochs,456600 steps,24 actual-loaded checkpoints,1200 episodes,120 vector attempts; queue exit0')
