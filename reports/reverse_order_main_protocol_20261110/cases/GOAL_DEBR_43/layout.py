from pathlib import Path
import json
R=Path(__file__).resolve().parent
ARMS='S'
SEED=43
NAME='REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering04'
ROOT_PRIVATE=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering04')/'unused_root_payload'
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering04')
OUTPUT={'S':Path('/data/mlr_ifa/experiments/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering04/GOAL_DEBR_43/S')}
VOLUME_BOUND={'/root':0,'/data':14*1024**3}
_counts=json.loads((R/'INPUT_COUNTS.json').read_text())
POP={int(k):v for k,v in _counts['population'].items()}
VALID={int(k):v for k,v in _counts['valid'].items()}
OLD={int(k):v for k,v in _counts['old_valid'].items()}
STAGE_STARTS=[0, 19700, 38600, 61900]
MEMORY_PER_TASK=128
ANCHOR_BATCH=4
RETAIN_WEIGHT=.1
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
def memory_path(arm,task):return OUTPUT[arm]/'memory'/('task%d.pth'%task)
