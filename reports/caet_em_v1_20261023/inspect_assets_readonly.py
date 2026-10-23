import os,json,hashlib,subprocess
from pathlib import Path
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
files={}
for name in ['libero/lifelong/models/policy_head.py','libero/lifelong/models/modules/rgb_modules.py','libero/lifelong/algos/mlr.py','coreset_exp/common.py','coreset_exp/train.py']:
 p=S/name;files[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(dict(status='READ_ONLY',source=str(S),config=json.loads((P/'resolved_eval.json').read_text()),files=files,storage={str(t):json.loads((P/'ra_coreset/storage'/('task%d.json'%t)).read_text())['selected'][:2] for t in range(6,10)},counts={str(t):len(list((P/'features'/str(t)).glob('*.pt'))) for t in range(6,10)},disk={p:dict(root_free=os.statvfs(p).f_bfree*os.statvfs(p).f_frsize,ordinary_free=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize) for p in ['/','/data']},gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),processes=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True))))
