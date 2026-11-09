SPEC={'OBJECT_41': {'parent': '/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc', 'align_parent': '/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01', 'align_source': '/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source'}, 'OBJECT_42': {'parent': '/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_OBJECT_seed42_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source'}, 'OBJECT_43': {'parent': '/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_OBJECT_seed43_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_43/source'}, 'GOAL_41': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source'}, 'GOAL_42': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed42_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_42/source'}, 'GOAL_43': {'parent': '/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed43_20261017', 'align_parent': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed43_20261022', 'align_source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_43/source'}}
from pathlib import Path
import json,hashlib,os,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={}
for case,s in SPEC.items():
 P=Path(s['parent']);A=Path(s['align_parent']);row=dict(parent_exists=P.exists(),parent_resolved=str(P.resolve()),align_parent=str(A),metadata={},inventory={})
 for rel in ['CONTRACT.json','MODAL_CONTRACT.json','resolved_random.json','resolved_eval.json','random/INITIAL_STATE.json','random/TRAINING_COMPLETE.json','INDEPENDENT_ACCEPTANCE.json','FEATURES.json']+[f'random/storage/task{t}.json' for t in range(6,10)]+[f'random/buffer_after_task{t}.json' for t in range(6,10)]:
  p=P/rel
  if p.exists():row['metadata'][rel]=dict(sha256=sha(p),text=p.read_text())
 for t in range(6,10):
  d=A/f'effects/current/task{t}';p=d/'AUDIT.json';j=json.loads(p.read_text());labels=d/'labels.pth'
  row['inventory'][str(t)]=dict(audit_sha256=sha(p),audit=j,labels_path=str(labels),labels_bytes=labels.stat().st_size,feature_count=len(list((P/'features'/str(t)).glob('*.pt'))))
 row['buffer_count']=len(list((P/'random/buffer').glob('*.pt')))
 out[case]=row
print(json.dumps(dict(status='READONLY_RANDOM_ALIGN_INPUT_INVENTORY',observed=time.time(),cases=out,new_optimizer=0,new_environments=0,new_gpu_calls=0,free_bytes={p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in ['/','/data']})))
