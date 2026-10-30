from pathlib import Path
ARMS='C'
SEED=43
SUITE='GOAL'
NAME='C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
CASE_PRIVATE=DATA_PRIVATE/'GOAL_43'
OUTPUT={a:CASE_PRIVATE/a for a in ARMS}
VOLUME_BOUND={'/root':0,'/data':18*1024**3}
STAGE_EPOCHS={6:100,7:200,8:200,9:200}
TOTAL_UPDATES=142200
POP={6: 997, 7: 1388, 8: 1848, 9: 3365}
VALID={6: 917, 7: 1285, 8: 1713, 9: 3204}
OLD={6: 0, 7: 476, 8: 889, 9: 1315}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return CASE_PRIVATE/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return CASE_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
