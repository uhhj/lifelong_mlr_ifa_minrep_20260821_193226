"""Archive only lightweight evidence after independent completion."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,time
O=Path('/data/mlr_ifa/experiments/E1O_coverage_order_seed42_20261009')
assert (O/'independent_acceptance.exit').read_text().strip()=='0'
a=json.loads((O/'INDEPENDENT_ACCEPTANCE.json').read_text());assert a['status']=='COMPLETED'
gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()
live=[p for p in processes if any(s in p for s in ['-m protective_e1.train','-m protective_e1.evaluate','protective_e1/pipeline.sh']) and 'bash -c' not in p]
assert not live
snapshot=dict(unix=time.time(),pipeline_exit=0,independent_exit=0,processes=live,
    gpu_query_exit=gpu.returncode,gpu_query_output=gpu.stdout+gpu.stderr,
    note='GPU unavailable at final audit; training and evaluation previously exited0. Audit used only CPU index arithmetic and file hashes; no fallback rollout.')
(O/'FINAL_PROCESS_SNAPSHOT.json').write_text(json.dumps(snapshot,indent=2)+'\n')
allowed={'.json','.jsonl','.csv','.log','.txt','.exit','.py','.sh','.yaml','.patch'}
paths=[p for p in O.iterdir() if p.is_file() and p.suffix in allowed]
for folder in ['order_logs','run_001','hydra_outputs']:
    paths.extend(p for p in (O/folder).rglob('*') if p.is_file() and p.suffix in allowed)
paths=sorted(set(paths));manifest={}
for p in paths:
    manifest[p.relative_to(O).as_posix()]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size)
(O/'FINAL_DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths.append(O/'FINAL_DOWNLOAD_MANIFEST.json')
with tarfile.open(O/'final_light_evidence.tar.gz','w:gz') as t:
    for p in paths:t.add(p,arcname=p.relative_to(O).as_posix())
p=O/'final_light_evidence.tar.gz'
print(json.dumps(dict(files=len(manifest),archive_members=len(paths),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
