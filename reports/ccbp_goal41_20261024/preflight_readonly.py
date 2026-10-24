import os,json,hashlib,subprocess,time
from pathlib import Path
O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022')
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(O/'CONTRACT.json')=='f556347a93aeeb17bfad177fc3088535bd03ae259e7198a40a805ac721b8bdc8'
c=json.loads((O/'CONTRACT.json').read_text());assert all(sha(S/n)==h for n,h in c['source_sha256'].items())
assert sha(O/'ra_coreset/models/mlrifa_8_model_ep99.pth')=='7c4e99ef40269d6d2169459d17818bb526afe288188372a7a6ebc2abf0247f3a'
files=['libero/lifelong/algos/mlr.py','libero/lifelong/models/brief.py','libero/lifelong/models/policy_head.py','coreset_exp/common.py','caet_v3_exp/train.py','caet_v3_exp/observe.py','caet_v3_exp/dataset.py','caet_v3_exp/model.py','coreset_exp/model.py']
src={n:(S/n).read_text() for n in files}
cfg=json.loads((O/'resolved_ra_coreset.json').read_text());m=json.loads((O/'ra_coreset/dataset_manifests/task9.json').read_text())
free={}
for root in ['/root','/data']:
 v=os.statvfs(root);free[root]=dict(bavail=v.f_bavail*v.f_frsize,bfree=v.f_bfree*v.f_frsize)
print(json.dumps(dict(status='READ_ONLY_PREFLIGHT',observed=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),capacity=free,source_hashes={n:sha(S/n) for n in files},sources=src,config=cfg,manifest_count=len(m['rows']),manifest_first=m['rows'][0],parent_acceptance=json.loads((O/'PROCESS_EXIT_ACCEPTANCE.json').read_text()))))
