from pathlib import Path
ARMS='S'
SEED=41
NAME='S_LAMBDA_R_OBJECT41_42_43_20261101_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
OUTPUT={a:DATA_PRIVATE/'L020_41'/a for a in ARMS}
VOLUME_BOUND={'/root':128*1024**2,'/data':12*1024**3}
POP={6: 1562, 7: 2261, 8: 3081, 9: 3620}
VALID={6: 1482, 7: 2164, 8: 2960, 9: 3475}
OLD={6: 0, 7: 764, 8: 1480, 9: 2236}
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=0.2
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)

SUITE="OBJECT"
LAMBDA_GROUP='L020'
