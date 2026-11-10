from pathlib import Path
ARMS='A'
SUITE='OBJECT'
SEED=41
NAME='S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1'
WORKSPACE=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/S_STRIPPED_MLR_IFA_ALIGN_OBJECT_GOAL_20261110_v1')
OUTPUT={'A':DATA_PRIVATE/'OBJECT_41'/'A'}
POP={6: 1562, 7: 2261, 8: 3081, 9: 3620}
VALID={6: 1482, 7: 2144, 8: 2924, 9: 3426}
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
