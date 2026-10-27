from pathlib import Path
ARMS='C'
SEED=41
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
CASE_PRIVATE=DATA_PRIVATE/'OBJECT_41'
OUTPUT={a:CASE_PRIVATE/a for a in ARMS}
# Conservative proposed bounds, must pass fresh remote preflight before dispatch.
VOLUME_BOUND={'/root':0,'/data':6*1024**3}
POP={6: 1562, 7: 2261, 8: 3081, 9: 3620}
VALID={6: 1482, 7: 2164, 8: 2960, 9: 3475}
OLD={6: 0, 7: 764, 8: 1480, 9: 2236}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return CASE_PRIVATE/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return CASE_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
