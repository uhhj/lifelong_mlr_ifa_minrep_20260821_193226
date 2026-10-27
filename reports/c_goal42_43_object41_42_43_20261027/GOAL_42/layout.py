from pathlib import Path
ARMS='C'
SEED=42
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
CASE_PRIVATE=DATA_PRIVATE/'GOAL_42'
OUTPUT={a:CASE_PRIVATE/a for a in ARMS}
# Conservative proposed bounds, must pass fresh remote preflight before dispatch.
VOLUME_BOUND={'/root':0,'/data':6*1024**3}
POP={6: 997, 7: 1388, 8: 1848, 9: 3365}
VALID={6: 917, 7: 1288, 8: 1723, 9: 3215}
OLD={6: 0, 7: 479, 8: 899, 9: 1326}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return CASE_PRIVATE/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return CASE_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
