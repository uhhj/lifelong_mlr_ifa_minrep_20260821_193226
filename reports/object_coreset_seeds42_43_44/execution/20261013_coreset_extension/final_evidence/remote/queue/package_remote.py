import json,time,subprocess,hashlib,tarfile
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_extension_20261013')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert (Q/'queue.exit').read_text().strip()=='0'
for seed in (43,44):assert json.loads((Q/f'FINAL_HASH_{seed}.json').read_text())['status']=='PASS'
proc=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
active=[x for x in proc if (' -m coreset_exp.' in x or '/run_queue.sh' in x or '/coreset_exp/supervise.sh' in x or '/coreset_exp/pipeline.sh' in x) and not any(y in x for y in ('bash -c','ps -eo'))]
assert not active,active
gpu=subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
proof={'status':'PASS','queue_exit':0,'residual_processes':active,'unix':time.time(),'current_gpu_query_exit':gpu.returncode,'current_gpu_query_output':gpu.stdout,'gpu_note':'Current query is after accepted training/evaluation; do not rerun experiments or alter server driver.'}
files=[]
for seed in (42,43,44):
    O=Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
    assert (O/'pipeline.exit').read_text().strip()==(O/'supervisor.exit').read_text().strip()=='0'
    assert json.loads((O/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'
    assert json.loads((O/'SUMMARY.json').read_text())['formal_records']==400
    for p in O.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.yaml','.log','.exit','.patch') and 'monitor' not in p.relative_to(O).parts:
            files.append((p,f'seed{seed}/'+str(p.relative_to(O))))
(Q/'FINAL_PROCESS_ACCEPTANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
for p in Q.iterdir():
    if p.is_file() and p.suffix in ('.json','.md','.log','.exit','.py','.sh') and not p.name.startswith(('check_','final_check_','AUDIT_','audit_','launch_snapshot')):
        files.append((p,'queue/'+p.name))
manifest={name:sha(p) for p,name in files}
(Q/'FINAL_DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(Q/'THREE_SEED_LIGHT_EVIDENCE.tar.gz','w:gz') as t:
    for p,name in files:t.add(p,arcname=name)
    t.add(Q/'FINAL_DOWNLOAD_MANIFEST.json',arcname='FINAL_DOWNLOAD_MANIFEST.json')
print(json.dumps({'proof':proof,'files':len(files),'archive_sha256':sha(Q/'THREE_SEED_LIGHT_EVIDENCE.tar.gz'),'bytes':(Q/'THREE_SEED_LIGHT_EVIDENCE.tar.gz').stat().st_size},indent=2))
