from pathlib import Path
ARMS='BCNV'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1')
OUTPUT={a:(ROOT_PRIVATE if a in 'BN' else DATA_PRIVATE)/a for a in ARMS}
TEACHER_CACHE=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1/teacher_cache.pth')
PRIOR_ROOT=Path('/data/mlr_ifa/experiment_workspaces/CCBP_GOAL41_20261024_v1')
PER_VOLUME_BOUND=int(1.04*1024**3)
