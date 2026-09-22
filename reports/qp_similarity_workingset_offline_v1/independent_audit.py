"""Read-only independent result recomputation, executed after timing finishes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,statistics,time
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
P=Path(__file__).resolve().parent
cv=lambda a:Decimal.from_float(float(a))
def checks(A,b,x):
 with localcontext() as c:
  c.prec=80;n=len(b);xd=list(map(cv,x));zero=Decimal(0)
  r=[cv(b[i])-sum((cv(A[i,j])*xd[j] for j in range(n)),zero) for i in range(n)]
  parts=[max([zero]+r),max([zero]+[-v for v in xd]),max(abs(a*z) for a,z in zip(xd,r))]
  return max(parts),r
rows=list(csv.DictReader((P/'per_instance.csv').open()))
errors=[];tested=0;max_e=0.;refs={};began=time.time()
for done in sorted((P/'cases').glob('*/tau_*/COMPLETE.json')):
 info=json.loads(done.read_text());id=info['fixture']['id'];tau=info['tau'];cell=done.parent
 z=np.load(P/'inputs'/(id+'.npz'));A,b=z['A'],z['b'];n=len(b)
 if id not in refs:
  ref=np.load(P/'references'/id/'reference.npz')['x'];rk,_=checks(A,b,ref)
  refs[id]=(ref,rk)
 ref,rk=refs[id]
 for arm in ['FULL_WS','SIM_ADD','RANDOM_ADD_0','RANDOM_ADD_1','RANDOM_ADD_2']:
  directory=cell/arm;meta=json.loads((directory/'repeat00.json').read_text())
  x=np.load(directory/'candidate.npz')['x'];k,_=checks(A,b,x)
  residual=b-A@x
  p64=dict(primal_positive=float(np.maximum(residual,0).max()),nonnegative_violation=float(np.maximum(-x,0).max()),complementarity=float(np.abs(x*residual).max()))
  p64['kkt']=max(p64.values())
  if p64!=meta['float64']:errors.append([id,tau,arm,'FLOAT64_RECOMPUTATION',p64,meta['float64']])
  if str(k)!=meta['decimal80']['kkt']:errors.append([id,tau,arm,'KKT_RECOMPUTATION',str(k),meta['decimal80']['kkt']])
  with localcontext() as c:
   c.prec=80;dx=[cv(v)-cv(w) for v,w in zip(x,ref)]
   q=sum((dx[i]*cv(A[i,j])*dx[j] for i in range(n) for j in range(n)),Decimal(0))
   e=q.sqrt() if q>=0 else Decimal('Infinity')
  max_e=max(max_e,float(e))
  expected=(p64['kkt']<=1e-8 and k<=Decimal('1e-8') and rk<=Decimal('1e-12') and e<=Decimal('1e-5'))
  if bool(meta['full_kkt_pass'] and meta['validation']['update_budget_pass'])!=expected:
   errors.append([id,tau,arm,'CERTIFICATION_DECISION'])
  repeats=[json.loads((directory/('repeat%02d.json'%i)).read_text()) for i in range(10)]
  med=statistics.median(r['total_seconds'] for r in repeats)
  for r in repeats:
   subtotal=sum(r[key] for key in ['grouping_seconds','selection_seconds','construction_seconds','subqp_seconds','full_check_seconds'])
   if subtotal>r['total_seconds']+1e-8:errors.append([id,tau,arm,'TIMING_COMPONENT_SUM'])
  rr=[r for r in rows if r['fixture']==id and float(r['tau'])==tau and r['arm']==('RANDOM_ADD' if arm.startswith('RANDOM') else arm) and (not arm.startswith('RANDOM') or r['random_seed']==arm[-1])]
  if len(rr)!=1 or float(rr[0]['total_seconds_median'])!=med:errors.append([id,tau,arm,'TIMING_MEDIAN'])
  trace=json.loads((directory/'constraint_trace.json').read_text())
  for before,after in zip(trace,trace[1:]):
   if set(after['C'])!=set(before['C'])|set(before['added']):errors.append([id,tau,arm,'OUTER_SET'])
  if arm=='SIM_ADD':
   first=np.load(directory/'candidate.npz')['first_x'];fk,fr=checks(A,b,first)
   inactive=set(range(n))-set(meta['first']['C']);f64=b-A@first
   adds=[i for i in inactive if fr[i]>Decimal('1e-8') or f64[i]>1e-8]
   if len(adds)!=meta['first']['missing_violation_count']:errors.append([id,tau,arm,'FIRST_MISSED_COUNT'])
  tested+=1
out=dict(status='PASS' if not errors else 'FAIL',candidates=tested,reference_count=len(refs),
 errors=errors,max_E_update_all_arms=max_e,reference_KKT={k:str(v[1]) for k,v in refs.items()},
 seconds=time.time()-began,independent_checker='Decimal.from_float,80 digits; no production checker or summary function called',
 training_active=False,formal_solver_modified=False,model_writeback=False,GPU_required=False)
p=P/'INDEPENDENT_AUDIT.json';t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(out,indent=2,allow_nan=False));os.replace(t,p)
print(json.dumps({k:v for k,v in out.items() if k!='reference_KKT'},indent=2))
