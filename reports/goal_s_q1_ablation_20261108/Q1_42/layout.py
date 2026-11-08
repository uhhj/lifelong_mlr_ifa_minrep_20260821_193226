from pathlib import Path
ARMS='S'
SEED=42
NAME='S_Q1_GOAL41_42_43_20261108_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
OUTPUT={a:DATA_PRIVATE/'Q1_42'/a for a in ARMS}
VOLUME_BOUND={'/root':128*1024**2,'/data':4*1024**3}
POP={6: 997, 7: 1388, 8: 1848, 9: 3365}
VALID={6: 917, 7: 1288, 8: 1723, 9: 3215}
OLD={6: 0, 7: 479, 8: 899, 9: 1326}
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=0.1
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)

SUITE="GOAL"
LAMBDA_GROUP='Q1'
SUPPORT_CALIBRATION='DISABLED_FIXED_ONE'
