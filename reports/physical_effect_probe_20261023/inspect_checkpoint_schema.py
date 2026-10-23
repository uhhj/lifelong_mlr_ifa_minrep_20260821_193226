import json,torch
from pathlib import Path
P=Path('/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth')
r=torch.load(P,map_location='cpu');S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
txt=(S/'libero/lifelong/utils.py').read_text();i=txt.index('def torch_load_model');txt=txt[i:i+2200]
print(json.dumps(dict(keys=list(r),state_keys=list(r['state_dict'])[:20],loader_source=txt,server_writes=0,forwards=0)))
