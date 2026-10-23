import os,json,subprocess,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03')
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text());bad=[p for p,h in c['files'].items() if sha(p)!=h]
cfg=json.loads((P/'resolved_eval.json').read_text())
print(json.dumps(dict(status='READONLY',old_exit=(Q/'controller.exit').read_text().strip(),bad=bad,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True),
 active_gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),
 disk={p:dict(root=os.statvfs(p).f_bfree*os.statvfs(p).f_frsize,ordinary=os.statvfs(p).f_bavail*os.statvfs(p).f_frsize) for p in ('/data','/')},
 policy_source=(S/'libero/lifelong/models/complete_policy.py').read_text(),
 policy_config=cfg.get('policy'),data_config=cfg.get('data'),old_contract_sha256=sha(Q/'CONTRACT.json'))))
