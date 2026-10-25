from pathlib import Path
ARMS='CRG'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/CCBP_V3_FULL_GOAL41_20261025_v1')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/CCBP_V3_FULL_GOAL41_20261025_v1')
OUTPUT={a:(ROOT_PRIVATE if a in 'CR' else DATA_PRIVATE)/a for a in ARMS}
# Conservative proposed bounds, must pass fresh remote preflight before dispatch.
VOLUME_BOUND={'/root':int(1.10*1024**3),'/data':int(1.40*1024**3)}
POP={6:997,7:1388,8:1848,9:3365}
VALID={6:917,7:1286,8:1723,9:3218}
OLD={6:0,7:477,8:899,9:1329}
def model_path(arm,stage):
 if stage==9:return OUTPUT[arm]/'stage9.pth'
 return (ROOT_PRIVATE if (arm,stage)==('C',6) else DATA_PRIVATE)/'stage_models'/arm/('stage%d.pth'%stage)
def teacher_path(arm,stage):return DATA_PRIVATE/'teachers'/arm/('stage%d.pth'%stage)
