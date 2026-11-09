SPEC={'OBJECT_41': {'parent': '/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc', 'align_parent': '/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01', 'align_source': '/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source'}, 'OBJECT_42': {'parent': '/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_OBJECT_seed42_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source'}, 'OBJECT_43': {'parent': '/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_OBJECT_seed43_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_43/source'}, 'GOAL_41': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source'}, 'GOAL_42': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed42_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_42/source'}, 'GOAL_43': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed43_20261017', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed43_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_43/source'}}
from pathlib import Path
import os,json,hashlib,time
Q=Path('/data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT_GOAL41_42_43_20261109_rerun_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text());assert sha(Q/'CONTRACT.json')=='fbbce2917380d1cde1025f0006b6fdd6bc7959f92226bb5588cc431ee0c8d96c'
assert all(sha(p)==h for p,h in c['files'].items())
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'S_Q1_' in cmd:procs.append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0]))
 except OSError:pass
sizes={}
for case,s in SPEC.items():
 P=Path(s['parent'])/'random';sizes[case]={str(p.relative_to(P)):p.stat().st_size for p in [*sorted((P/'models').glob('*.pth')),P/'recovery/latest.pth'] if p.exists()}
new=Q.parent/'MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1';private=Path('/data/mlr_ifa/experiments')/new.name
print(json.dumps(dict(status='READONLY_PREPARATION_RESOURCE_SNAPSHOT',observed=time.time(),current_q1_frozen_files_exact=len(c['files']),current_q1_contract=sha(Q/'CONTRACT.json'),current_q1_processes=procs,new_root_exists=new.exists(),new_private_root_exists=private.exists(),historical_private_sizes=sizes,free_bytes={p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in ['/','/data']},new_optimizer=0,new_environments=0,new_gpu_calls=0)))
