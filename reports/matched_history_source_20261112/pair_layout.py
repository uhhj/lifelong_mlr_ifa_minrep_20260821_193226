from pathlib import Path
import json

CASE_SPEC = json.loads(Path(__file__).with_name('CASE.json').read_text())
SEED = CASE_SPEC['seed']
CASE = '%s_%d' % (CASE_SPEC['suite'], SEED)
ARMS = 'ES'
TEACHERS = CASE_SPEC['teachers']
NAME = 'MATCHED_HISTORY_SOURCE_GOAL_OBJECT_20261112_v1'
ROOT_PRIVATE = Path('/root/mlr_ifa/experiments') / NAME
DATA_PRIVATE = Path('/data/mlr_ifa/experiments') / NAME
OUTPUT = {arm: DATA_PRIVATE / CASE / arm for arm in ARMS}
ACQUISITION = Path('/data/mlr_ifa/experiment_workspaces') / NAME / CASE
MATCHED_MEMORY = DATA_PRIVATE / CASE / 'matched_memory.pth'
COMMON_INITIAL = DATA_PRIVATE / CASE / 'COMMON_INITIAL.pth'
VOLUME_BOUND = {'/root': 0, '/data': 8 * 1024**3}
MEMORY_PER_TASK = 128
ANCHOR_BATCH = 4
RETAIN_WEIGHT = .1
STAGE = 9


def model_path(arm, stage):
    if stage < STAGE:
        return Path(TEACHERS[str(stage)]['path'])
    return OUTPUT[arm] / 'stage9.pth'


def acquisition_path(task):
    return DATA_PRIVATE / CASE / 'acquisition' / 'memory' / ('task%d.pth' % task)
