from pathlib import Path
ARMS='Q'
SEED=42
NAME='FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v2'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments')/NAME
DATA_PRIVATE=Path('/data/mlr_ifa/experiments')/NAME
OUTPUT={a:DATA_PRIVATE/'GOAL_42'/a for a in ARMS}
VOLUME_BOUND={'/root':0,'/data':8*1024**3}
POP={6: 997, 7: 1388, 8: 1848, 9: 3365}
VALID={6: 917, 7: 1288, 8: 1723, 9: 3215}
OLD={6: 0, 7: 479, 8: 899, 9: 1326}
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=.1
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)

TOTAL_UPDATES=76100
SUITE='GOAL'
