from pathlib import Path
ARMS='S'
SEED=43
NAME='S_LAMBDA_R_GOAL41_42_43_20261103_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
OUTPUT={a:DATA_PRIVATE/'L020_43'/a for a in ARMS}
VOLUME_BOUND={'/root':128*1024**2,'/data':12*1024**3}
POP={6: 997, 7: 1388, 8: 1848, 9: 3365}
VALID={6: 917, 7: 1285, 8: 1713, 9: 3204}
OLD={6: 0, 7: 476, 8: 889, 9: 1315}
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=0.2
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)

SUITE="GOAL"
LAMBDA_GROUP='L020'
