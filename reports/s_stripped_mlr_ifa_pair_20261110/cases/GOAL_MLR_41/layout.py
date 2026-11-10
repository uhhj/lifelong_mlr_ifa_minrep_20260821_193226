from pathlib import Path
ARMS='A'
SUITE='GOAL'
SEED=41
NAME='S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1'
WORKSPACE=Path('/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1')
OUTPUT={'A':DATA_PRIVATE/'GOAL_MLR_41'/'A'}
POP={6: 997, 7: 1388, 8: 1848, 9: 3365}
VALID={6: 917, 7: 1263, 8: 1681, 9: 3157}
def model_path(arm,stage):return OUTPUT[arm]/('stage%d.pth'%stage)
