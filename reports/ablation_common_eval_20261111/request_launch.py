from pathlib import Path
import json,subprocess,sys
root=Path('/data/mlr_ifa/experiment_workspaces/ABLATION_COMMON_DEBR_EVAL_20261111_v2')
sys.path.insert(0,str(root))
from job import launch
print(json.dumps(launch({'predecessor_final_audit_sha256': 'e5041ee35b10e70bd1cfbd3aca92010d2f5b42f2b6d38ad29d85a54a06d2c5f2', 'predecessor_publication': '6fa12be405179982cd20253728d14faa720a18da', 'remote_cpu_results': '/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05/final_cpu_20261112_155206/CPU_RESULTS.json', 'remote_cpu_results_sha256': '56013de46405a65c2af7c78de97ce19dcd02577200d59493b2da4a9a2d84bfc7', 'prepared_contract': '6cb37dfcc312ca8bd9597b673303e4ff7bbf1f2bd9953a636353a201b353ca13'})))
