"""Supply frozen evaluation inputs to the original DEBR rollout source."""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from common import ROOT, capacity, read as js, sha, verify_frozen, verify_inputs, write

NAME = os.environ['ABLATION_EVAL_CASE']
CASE = js(ROOT / 'SPEC.json')['evaluation_cases'][NAME]
R = ROOT / 'cases' / NAME
O = Path(CASE['config_root'])
SEED = CASE['seed']
# The original evaluator uses the S label to select its single policy slot.
# The case metadata identifies which historical model supplies that slot.
ARMS = 'S'
os.environ.update(CORESET_OUT=str(O), CORESET_SEED=str(SEED))
sys.path.insert(0, str(CASE['source_root']))

import numpy as np
import torch
from easydict import EasyDict
from libero.lifelong.utils import control_seed

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False


def model_path(arm, stage):
    assert arm == 'S'
    return Path(CASE['models'][str(stage)]['path'])


def verify():
    assert verify_frozen()['evaluation_cases'][NAME] == CASE
    verify_inputs(CASE)
