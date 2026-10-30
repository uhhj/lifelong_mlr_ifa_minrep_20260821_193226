from pathlib import Path
ARMS='C'
SEED=42
SUITE='OBJECT'
NAME='C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
CASE_PRIVATE=DATA_PRIVATE/'OBJECT_42'
OUTPUT={a:CASE_PRIVATE/a for a in ARMS}
VOLUME_BOUND={'/root':0,'/data':18*1024**3}
STAGE_EPOCHS={6:100,7:200,8:200,9:200}
TOTAL_UPDATES=195300
POP={6: 1562, 7: 2261, 8: 3081, 9: 3620}
VALID={6: 1482, 7: 2162, 8: 2962, 9: 3480}
OLD={6: 0, 7: 762, 8: 1482, 9: 2241}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return CASE_PRIVATE/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return CASE_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
