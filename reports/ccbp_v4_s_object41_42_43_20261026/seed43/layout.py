from pathlib import Path
ARMS='S'
SEED=43
NAME='CCBP_V4_S_OBJECT41_42_43_20261026_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
OUTPUT={a:DATA_PRIVATE/('seed%d'%SEED)/a for a in ARMS}
VOLUME_BOUND={'/root':0,'/data':6*1024**3}
POP={6: 1562, 7: 2261, 8: 3081, 9: 3620}
VALID={6: 1482, 7: 2161, 8: 2956, 9: 3468}
OLD={6: 0, 7: 761, 8: 1476, 9: 2229}
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=.1
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)
