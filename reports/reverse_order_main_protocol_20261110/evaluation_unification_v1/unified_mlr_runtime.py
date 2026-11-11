"""Evaluation-only adapter; no training methods, optimizer or acquisition."""
from pathlib import Path
import os,sys,json,hashlib,time,copy
import numpy as np
import torch
from easydict import EasyDict
Q=Path(__file__).resolve().parent.parent
R=Path(os.environ['REVERSE_CASE_ROOT'])
INPUT=json.loads((R/'INPUT.json').read_text())
assert INPUT['method']=='MLR'
O=Path(INPUT['input_root']);SEED=INPUT['seed'];ARMS=['MLR']
from order import install,ORIGINAL_TASKS
install()
from coreset_exp.common import sha,write
from libero.lifelong.utils import control_seed
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
def js(p):return json.loads(Path(p).read_text())
def model_path(arm,stage):
 assert arm=='MLR' and stage in range(6,10)
 return Path(INPUT['output_root'])/'models'/('mlrifa_%d_model_ep99.pth'%stage)
def verify():
 from reverse_common import verify_case
 m=js(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
 c=js(Q/'CONTRACT.json')
 assert m['id']=='COMMON_DEBR_EVALUATION_20261111_V1'
 assert c['evaluation_amendment']['manifest_sha256']==sha(Q/'EVALUATION_PROTOCOL_AMENDMENT.json')
 assert m['cases'][R.name]['training_contract_sha256']==sha(R/'CONTRACT.json')
 for name,want in m['evaluation_source_sha256'].items():assert sha(Q/name)==want,name
 return verify_case(R)
def capacity():
 from reverse_common import capacity as check
 return check()
