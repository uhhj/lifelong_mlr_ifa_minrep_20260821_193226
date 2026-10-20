"""Read-only progress snapshot; no ML imports."""
import json, subprocess, time, shutil
from server_runner import N, paths, js, sha, verify, workers
verify()
out=dict(time=time.time(),workers=workers(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
         data_free_GiB=shutil.disk_usage('/data').free/1024**3,variants={})
for name in ('RUN_STATUS.json','controller.started','controller.exit','supervisor.exit','BRANCH_DECISION.json','FINAL_ACCEPTANCE.json','PROCESS_EXIT_ACCEPTANCE.json','BLOCKED.json'):
    p=N/name
    if p.exists():out[name]=js(p) if name.endswith('.json') or name.endswith('.started') else p.read_text()
for v in ('v3','v3a'):
    W,O=paths(v)
    if not O.exists():continue
    d=dict(epochs=0,optimizer_steps=0,episodes=0,completed_vectors=0)
    for name in ('RUN_STATUS.json','PREFLIGHT_ACCEPTANCE.json','PROCESS_EXIT_ACCEPTANCE.json','ADMISSION_FAILURE.json'):
        if (O/name).exists():d[name]=js(O/name)
    if (O/'epochs.json').exists():
        rows=js(O/'epochs.json');d.update(epochs=len(rows),optimizer_steps=rows[-1]['optimizer_steps'],last_epoch=rows[-1],
                                         mean_epoch_seconds=sum(x['seconds'] for x in rows)/len(rows))
    ev=[json.loads(l) for p in (O/'evaluation').glob('trials*.jsonl') for l in p.read_text().splitlines() if l.strip()]
    d['episodes']=len(ev)
    events=[json.loads(l) for p in (O/'evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines() if l.strip()]
    d['completed_vectors']=sum(r['event']=='vector_attempt_complete' for r in events)
    d['technical_exceptions']=sum(r['event']=='technical_exception' for r in events)
    for p in list(O.glob('*.exit'))+list((O/'preflight').glob('*.exit')):d[str(p.relative_to(O))]=p.read_text().strip()
    d['recent_logs']={str(p.relative_to(O)):dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=p.read_text(errors='replace')[-1500:]) for p in list(O.glob('*.log'))+list((O/'preflight').glob('*.log'))}
    if (O/'CONTRACT.json').exists():
        c=js(O/'CONTRACT.json');d['frozen_hashes']=all(sha(p)==h for p,h in c['files'].items())
        assert d['frozen_hashes']
    out['variants'][v]=d
print(json.dumps(out,indent=2))
