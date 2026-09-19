from pathlib import Path
import shutil, json, textwrap

ROOT=Path('/data/lifelong_mlr_ifa_minrep_20260912')
BASE=Path('/tmp/qp_solver_recovery_v2')
if BASE.exists(): shutil.rmtree(BASE)
BASE.mkdir()
for pkg in ['sample_qp_lifelong_v1_continuation_v2','sample_qp_optimized_v1_continuation_v2']:
    shutil.copytree(ROOT/pkg, BASE/pkg)
src=BASE/'sample_qp_optimized_v1_continuation_v2/solver.py'
s=src.read_text().replace('import numpy as np','import numpy as np\nimport json\nfrom pathlib import Path')
start=s.index('            if err>1e-8:')
end=s.index('            lam[valid]=x*u_norm/scale', start)
old=s[start:end]
new="""              if err>1e-8:
                  for extra in range(20000):
                      for i in range(len(x)):
                          x[i]=max(0.,x[i]+(b[i]-float(A[i]@x))/A[i,i])
                      residual=b-A@x
                      err=max(float(np.maximum(residual,0).max()),float(np.abs(x*residual).max()))
                      if err<=1e-11:break
                  sweep += extra+1
              nit+=sweep+1
              primal=b-A@x
              kkt=max(float(np.maximum(primal,0).max()),float(np.maximum(-x,0).max()),float(np.abs(x*primal).max()))
              if kkt>1e-8:
                  out=Path('/data/lifelong_mlr_ifa_minrep_20260912/reports/qp_solver_recovery_v2')
                  out.mkdir(parents=True,exist_ok=True)
                  np.savez(out/'F2_before_extended.npz',K=K,f=f,u_norm=np.asarray(u_norm),valid=valid,scale=scale,A=A,b=b,x=x,primal=primal,kkt=np.asarray(kkt),sweep=np.asarray(sweep),nit=np.asarray(nit))
                  (out/'F2_before_extended.json').write_text(json.dumps({'stage':'after_existing_path','kkt':kkt,'sweep':int(sweep),'nit':int(nit),'shape':list(K.shape)},indent=2))
                  for ext in range(100000):
                      for i in range(len(x)):
                          x[i]=max(0.,x[i]+(b[i]-float(A[i]@x))/A[i,i])
                      residual=b-A@x
                      err=max(float(np.maximum(residual,0).max()),float(np.abs(x*residual).max()))
                      if err<=1e-8:break
                  sweep += ext+1
              primal=b-A@x
              kkt=max(float(np.maximum(primal,0).max()),float(np.maximum(-x,0).max()),float(np.abs(x*primal).max()))
              if kkt<=1e-8:
                  np.savez('/data/lifelong_mlr_ifa_minrep_20260912/reports/qp_solver_recovery_v2/F2_final.npz',K=K,f=f,u_norm=np.asarray(u_norm),valid=valid,scale=scale,A=A,b=b,x=x,primal=primal,kkt=np.asarray(kkt),sweep=np.asarray(sweep),nit=np.asarray(nit))
              assert kkt<=1e-8,('dual QP did not meet frozen KKT after bounded extended polish',fit.message,kkt)
"""
new=textwrap.indent(textwrap.dedent(new), '            ')
if old not in s: raise RuntimeError('solver pattern not found')
src.write_text(s.replace(old,new))
c=BASE/'sample_qp_lifelong_v1_continuation_v2/common.py'; t=c.read_text()
t=t.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path('/data/lifelong_mlr_ifa_minrep_20260912')")
t=t.replace("OUT = ROOT/'reports/sample_qp_lifelong_v1_continuation_v2'", "OUT = ROOT/'reports/qp_solver_recovery_v2/runtime'")
t=t.replace("RUN = ROOT/'runs/sample_qp_lifelong_v1_continuation_v2/seed46/sample_qp'", "RUN = ROOT/'runs/qp_solver_recovery_v2/seed46/sample_qp'")
c.write_text(t)
out=ROOT/'reports/qp_solver_recovery_v2/runtime'; out.mkdir(parents=True,exist_ok=True)
orig=ROOT/'reports/sample_qp_lifelong_v1_continuation_v2'
for p in ['PROTOCOL_FROZEN.json','PREFLIGHT_COMPLETE.json','INITIALIZATION_EXPECTED.json','PROTOCOL_CONTINUATION_FROZEN.json','PROTOCOL_PARENT.json']:
    if (orig/p).exists(): shutil.copy2(orig/p,out/p)
shutil.copytree(orig/'plans',out/'plans',dirs_exist_ok=True)
run=ROOT/'runs/qp_solver_recovery_v2/seed46/sample_qp'; run.mkdir(parents=True,exist_ok=True)
oldrun=ROOT/'runs/sample_qp_lifelong_v1_continuation_v2/seed46/sample_qp'
shutil.copy2(oldrun/'ATTEMPT_000.json',run/'ATTEMPT_000.json')
shutil.copy2(oldrun/'RECOVERY_INDEX.json',run/'RECOVERY_INDEX.json')
print(json.dumps({'base':str(BASE),'out':str(out),'run':str(run),'recovery_index':json.loads((run/'RECOVERY_INDEX.json').read_text())},indent=2))
