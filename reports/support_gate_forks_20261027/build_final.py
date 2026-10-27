"""Build external CPU-only final wrapper; does not invoke it."""
from pathlib import Path
import json,ast
R=Path(__file__).resolve().parent;P=R.parent/'20261027_c_goal42_43_object41_42_43';ROOT='/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1'
old=(P/'final_GOAL_42_readonly.py').read_text(encoding='utf8')
imports=old[old.index('import numpy as np'):old.index('def verify():')]
policy=old[old.index('def exact_class('):old.index('# Appended to external native')]
scripts={}
for name,c in json.loads((R/'CASES.json').read_text(encoding='utf8')).items():
    b=json.loads((R/name/'BINDING.json').read_text(encoding='utf8'))
    s="import os,sys,ast,json,hashlib,time,contextlib\nfrom pathlib import Path\nos.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')\n"
    s+=f"Q=Path({(ROOT+'/'+name)!r});O=Path({b['parent']!r});S=Path({b['source']!r})\nsys.path.insert(0,str(Q))\nfrom layout import *\n"
    s+=imports+policy+(R/'final_case_body.py').read_text(encoding='utf8');ast.parse(s,feature_version=(3,8));scripts[name]=s
    (R/('final_'+name+'_readonly.py')).write_text(s,encoding='utf8')
template=(P/'final_template.py').read_text(encoding='utf8').replace('/data/mlr_ifa/experiment_workspaces/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1',ROOT)
(R/'final_readonly.py').write_text('SCRIPTS='+ascii(scripts)+'\n'+template,encoding='utf8');ast.parse((R/'final_readonly.py').read_text(encoding='utf8'),feature_version=(3,8))
print('External CPU wrapper compiled, no final execution or acceptance claimed.')
