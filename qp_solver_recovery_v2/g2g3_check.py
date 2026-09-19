import json, numpy as np
from pathlib import Path
from scipy.optimize import minimize
from qp_tolerance_calibration_v1.adapter import load_fixture,residual_parts,mp_residual
root=Path('/data/lifelong_mlr_ifa_minrep_20260912'); out=root/'reports/qp_solver_recovery_v2'
z=load_fixture(out/'F2_before_extended.npz'); A=z['A']; b=z['b']; x0=z['x']
obj=lambda x: .5*float(x@A@x)-float(b@x)
jac=lambda x:A@x-b
fit=minimize(obj,x0,jac=jac,bounds=[(0,None)]*len(x0),method='SLSQP',options={'ftol':1e-14,'maxiter':100000,'disp':False})
x=np.maximum(fit.x,0); rr=residual_parts(A,b,x); hp=mp_residual(A,b,x,80)
assets={n:len(list(root.rglob(n))) for n in ['*K*.npz','*qp*.npz','*reference*.npz']}
result={'reference_solver':'scipy SLSQP independent restart from captured F2 solution','success':bool(fit.success),'message':str(fit.message),'nit':int(fit.nit),'objective':obj(x),'residual':rr,'high_precision':hp,'real_asset_counts':assets,'writeback_parity':'NOT_RUN_NO_H_OR_MODEL_STATE'}
(out/'G2G3_CHECK.json').write_text(json.dumps(result,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else o))
print(json.dumps(result,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else o))
