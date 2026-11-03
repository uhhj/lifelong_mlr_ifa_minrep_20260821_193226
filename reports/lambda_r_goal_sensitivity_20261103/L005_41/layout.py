from pathlib import Path
ARMS='S'
SEED=41
NAME='S_LAMBDA_R_GOAL41_42_43_20261103_v1'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
OUTPUT={a:DATA_PRIVATE/'L005_41'/a for a in ARMS}
VOLUME_BOUND={'/root':128*1024**2,'/data':12*1024**3}
POP={6:997,7:1388,8:1848,9:3365}
VALID={6:917,7:1286,8:1723,9:3218}
OLD={6:0,7:477,8:899,9:1329}
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=0.05
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)

SUITE="GOAL"
LAMBDA_GROUP='L005'
