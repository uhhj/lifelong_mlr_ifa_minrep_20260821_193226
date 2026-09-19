import importlib.util
import sys
from pathlib import Path
sys.path.insert(0,'/data/lifelong_mlr_ifa_minrep_20260912')
sys.path.insert(0,'/data/lifelong_mlr_ifa_minrep_20260912/libero')
p=Path('/tmp/qp_solver_recovery_v2/sample_qp_optimized_v1_continuation_v2/solver.py')
spec=importlib.util.spec_from_file_location('recovery_solver_impl',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig=__import__('sample_qp_optimized_v1_continuation_v2.solver',fromlist=['solve'])
orig.solve=m.solve
