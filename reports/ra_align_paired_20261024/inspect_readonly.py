import os,sys,json,hashlib,subprocess,time
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def emit(d):print(json.dumps(d,separators=(',',':')),flush=True)
entries=[]
for e in SPEC['seeds']:
 O=Path(e['ra_output']);assert sha(O/'CONTRACT.json')==e['ra_contract_sha256']
 c=js(O/'CONTRACT.json')
 a=js(O/'ra_coreset/storage/task8.json')['selected'];b=js(Path(e['output'])/'ra_coreset/storage/task8.json')['selected']
 assert [(x['destination'],x['sha256']) for x in a]==[(x['destination'],x['sha256']) for x in b]
 entries.append({'seed':e['seed'],'contract':c,'ra_cp_sha256':{str(t):sha(O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%t)) for t in (8,9)},'metadata':{'storage_sha256':sha(O/'ra_coreset/storage/task8.json')},'exits':{p.name:p.read_text() for p in O.glob('*.exit')},'acceptance':js(O/'PROCESS_EXIT_ACCEPTANCE.json')})
S=Path(SPEC['seeds'][0]['source']);names=['libero/lifelong/models/complete.py','libero/lifelong/metric.py','libero/lifelong/algos/basic.py','coreset_exp/model.py','coreset_exp/common.py','coreset_exp/evaluate.py','coreset_exp/eval_metric.py']
sources={n:(S/n).read_text() for n in names if (S/n).exists()}
configs={p.name:js(p) for p in Path(SPEC['seeds'][0]['output']).glob('resolved*json')}
emit({'kind':'inspect','entries':entries,'sources':sources,'configs':configs,'files':[str(p.relative_to(S)) for p in (S/'libero/lifelong/models').glob('*.py')],'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),'disk':{p:{'bfree':os.statvfs(p).f_bfree*os.statvfs(p).f_frsize,'bavail':os.statvfs(p).f_bavail*os.statvfs(p).f_frsize} for p in ['/data','/root']},'time':time.time(),'forwards':0,'environments':0})
