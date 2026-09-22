import csv,json,os,shutil,subprocess,time
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
import scipy
from scipy.optimize import minimize
from .fallback import run,parts,OPTIONS
from qp_v3_failure_capture_v1.capture import write_json
from sample_qp_lifelong_v1_continuation_v3 import common as c
ROOT=c.ROOT;OUT=ROOT/'reports/qp_lbfgsb_kkt_driven_v1';SOURCE=ROOT/'reports/qp_v3_failure_capture_v1/t8_e11_b258'

def mp(A,b,x):
    with localcontext() as ctx:
        ctx.prec=80;cv=lambda v:Decimal.from_float(float(v));xd=list(map(cv,x));bd=list(map(cv,b));zero=Decimal(0)
        p=[bd[i]-sum((cv(A[i,j])*xd[j] for j in range(len(x))),zero) for i in range(len(x))]
        vals=dict(primal_positive=max([zero]+p),nonnegative_violation=max([zero]+[-v for v in xd]),complementarity=max(abs(xd[i]*p[i]) for i in range(len(x))))
        vals['kkt']=max(vals.values());return {k:str(v) for k,v in vals.items()}

def save_npz(path,**arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.npz')
    with tmp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    with np.load(path,allow_pickle=False) as z:assert set(z.files)==set(arrays)

def write_result(name,A,b,x):
    out=OUT/name;out.mkdir(exist_ok=True)
    accepted,final,trace,meta=run(A,b,x)
    hp=mp(A,b,accepted);meta['selected_decimal80']=hp;meta['final_decimal80']=mp(A,b,final);meta['passed']=meta['accepted'] and Decimal(hp['kkt'])<=Decimal('1e-8')
    meta['formal_training_resumed']=False
    write_json(out/'RESULT.json',meta)
    with (out/'iterations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['iteration','objective','dx_norm','primal_positive','nonnegative_violation','complementarity','kkt']);w.writeheader();w.writerows(trace)
    save_npz(out/'candidate.npz',A=A,b=b,x_standard=x,x=accepted,x_final=final,first_pass_x=accepted if meta['first_pass_iteration'] is not None else np.empty(0))
    return meta

def main():
    OUT.mkdir(exist_ok=True)
    protocol=dict(candidate='LBFGSB_KKT_DRIVEN_V1',options=dict(OPTIONS,maxcor='len(x)'),kkt_tolerance=1e-8,qp='min x>=0: .5*x.T@A@x-b.T@x',standard_path='unchanged SLSQP + 10000 base + 20000 continuation',selection_priority=['callback first_pass_x','final_x if no first pass'],callback_does_not_stop_solver=True,normalization_unchanged=True,lambda_mapping='lam[valid]=x*u_norm/scale',reference_precision='80 decimal digits from exact binary64 values',scipy=scipy.__version__,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),formal_training_resumed=False)
    if (OUT/'FALLBACK_PROTOCOL.json').exists():assert c.read(OUT/'FALLBACK_PROTOCOL.json')==protocol
    else:write_json(OUT/'FALLBACK_PROTOCOL.json',protocol)
    paths=[SOURCE/n for n in ['before_fallback.npz','after_fallback.npz','FAILURE_CONTEXT.json','FAILURE_BATCH_INPUT.json','UPDATE_INPUT_FINGERPRINT.json','PRE_FAILURE_STATE.json']]+[Path('/root/qp_v3_failure_capture_v1/PRE_FAILURE_STATE.pt')]
    manifest={str(p):dict(bytes=p.stat().st_size,sha256=c.sha(p)) for p in paths}
    z=np.load(SOURCE/'before_fallback.npz');a=np.load(SOURCE/'after_fallback.npz');checks={k:np.array_equal(z[k],a[k]) for k in ['K','f','A','b','u_norm','valid','scale']}
    checks['shape']=z['A'].shape==(64,64);checks['finite']=all(np.isfinite(o[k]).all() for o in (z,a) for k in ['A','b','K','f','x','scale','u_norm'])
    p0=parts(z['A'],z['b'],z['x']);p1=parts(a['A'],a['b'],a['x'])
    checks['before_kkt']=p0['kkt']==1.0093558544532755e-8;checks['after_kkt']=p1['kkt']==1.547045713756745e-8
    original=json.loads(str(a['metadata_json']))['fallback'];checks['nit_one']=original['nit']==1;checks['relative_reduction']='REL_REDUCTION_OF_F_' in original['message']
    state_meta=c.read(SOURCE/'PRE_FAILURE_STATE.json');checks['state_sha']=manifest[str(paths[-1])]['sha256']==state_meta['sha256']
    checks['position']=(state_meta['batch'],state_meta['global_step'],state_meta['optimizer_step'],state_meta['scheduler_step'])==(258,42057,3657,3657)
    validation=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,assets=manifest,before=p0,after=p1,before_decimal80=mp(z['A'],z['b'],z['x']),after_decimal80=mp(a['A'],a['b'],a['x']),original_fallback=original,formal_training_resumed=False)
    write_json(OUT/'B258_ASSET_VALIDATION.json',validation);assert all(checks.values()),checks
    fixture=OUT/'frozen_b258';fixture.mkdir(exist_ok=True)
    for p in paths[:-1]:
        dst=fixture/p.name
        if dst.exists():assert c.sha(dst)==c.sha(p)
        else:shutil.copyfile(p,dst);assert c.sha(dst)==c.sha(p)
    A=z['A'];b=z['b'];x=z['x'];obj=lambda q:.5*float(q@A@q)-float(b@q);t=time.perf_counter()
    opts=dict(maxiter=100000,maxfun=200000,ftol=1e-15,gtol=1e-12,maxls=50,maxcor=len(x))
    fit=minimize(obj,x.copy(),jac=lambda q:A@q-b,bounds=[(0,None)]*len(x),method='L-BFGS-B',options=opts)
    cx=np.maximum(fit.x,0);pc=parts(A,b,cx)
    control=dict(options=opts,nit=int(fit.nit),message=str(fit.message),success=bool(fit.success),parts=pc,decimal80=mp(A,b,cx),seconds=time.perf_counter()-t,exact_previous_x=np.array_equal(cx,a['x']),formal_training_resumed=False)
    control['status']='PASS' if control['exact_previous_x'] and control['nit']==1 and pc['kkt']>1e-8 and 'REL_REDUCTION' in control['message'] else 'FAIL'
    write_json(OUT/'CURRENT_V3_CONTROL.json',control);assert control['status']=='PASS'
    result=write_result('B258',A,b,x);print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    write_json(OUT/'STATUS.json',dict(stage='B258_OFFLINE_PASS' if result['passed'] else 'B258_CANDIDATE_FAILED',formal_training_resumed=False))

if __name__=='__main__':main()
