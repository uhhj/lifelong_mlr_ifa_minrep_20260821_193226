"""Generate external CPU wrapper from previously reviewed native class extraction."""
from pathlib import Path
import ast
R=Path(__file__).resolve().parent;old=(R.parent/'20261024_ccbp_v2/final_readonly.py').read_text(encoding='utf8')
prefix=old[:old.index("manifest=js(O/'ra_coreset/dataset_manifests/task9.json')")]
prefix=prefix.replace("Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1')","Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02')")
start=prefix.index('OUTPUT=');end=prefix.index('import numpy',start)
prefix=prefix[:start]+"sys.path.insert(0,str(Q))\nfrom layout import *\n"+prefix[end:]
suffix=(R/'final_body.py').read_text(encoding='utf8');text=prefix+suffix;ast.parse(text);(R/'final_readonly.py').write_text(text,encoding='utf8')
print('AST ONLY; external final CPU wrapper generated, not executed')
