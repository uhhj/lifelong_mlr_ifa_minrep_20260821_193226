from pathlib import Path
ARMS='A'
SUITE='OBJECT'
SEED=43
NAME='S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1'
WORKSPACE=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1')
OUTPUT={'A':DATA_PRIVATE/'OBJECT_43'/'A'}
POP={6: 1562, 7: 2261, 8: 3081, 9: 3620}
VALID={6: 1482, 7: 2140, 8: 2913, 9: 3407}
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
