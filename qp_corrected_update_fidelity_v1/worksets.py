"""Conditional Phase 2. Selection only changes initial subsets, never QP semantics."""
import copy,gc,time,traceback
import numpy as np,pandas as pd,torch
from scipy.sparse.linalg import LinearOperator,eigsh
from sample_qp_optimized_v1_continuation_v4.solver import solve,dual_kkt
from . import engine
from .engine import g,native,inner,Workspace
from .common import *
from .analysis import clean,stats

def canonical(refs):return sorted(range(len(refs)),key=lambda i:(refs[i]['task_id'],str(refs[i]['trajectory_id']),refs[i]['window_index'],refs[i]['origin']))
def normalize(a):return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-300)
def fps(v,k,order):
 v=normalize(np.asarray(v,float));chosen=[order[0]];distance=1-v@v[chosen[0]]
 while len(chosen)<k:
  left=[i for i in order if i not in chosen];pick=max(left,key=lambda i:distance[i]);chosen.append(pick);distance=np.minimum(distance,1-v@v[pick])
 return sorted(chosen)
def stride(refs,k):
 order=canonical(refs);return sorted(order[int((j+.5)*len(order)/k)] for j in range(k))

def fingerprint(algo,ds,refs,destination):
 from gn_functional_geometry_v1.engine import Engine
 begin=time.perf_counter();before=g.capture(algo);algo.policy.eval();model=g.module(algo);zs=[];latents=[]
 try:
  with torch.no_grad():
   for batch in engine.refs_batches(algo,ds,refs):
    h=batch['feature'].squeeze(1);zs.append(model.temporal_encode(h)[:,-1,:]);latents.append(h.mean(tuple(range(1,h.ndim-1))))
  z=torch.cat(zs).double();latent=torch.cat(latents).double().cpu().numpy();latent_seconds=time.perf_counter()-begin
  metric=read(ROOT/'reports/gn_functional_geometry_v1/ACTION_METRIC.json');head=copy.deepcopy(model.policy_head).double().eval()
  e=Engine(head,{'old_build':z},{},torch.tensor(metric['diagonal'],device=z.device,dtype=torch.float64));P=e.theta.numel()
  op=LinearOperator((P,P),matvec=lambda v:e.cv(v).cpu().numpy(),dtype=np.float64)
  vals,U=eigsh(op,k=32,ncv=72,tol=1e-6,maxiter=500,v0=np.random.RandomState(2026092303).normal(size=P));ix=np.argsort(vals)[::-1];vals=vals[ix];U=U[:,ix].astype(np.float32).astype(np.float64)
  orth=float(np.max(abs(U.T@U-np.eye(32))));q=[];res=[]
  for k in range(32):
   u=torch.as_tensor(U[:,k],device=z.device,dtype=torch.float64);q.append((e.jv(u).square()*e.weight).sum(-1).cpu().numpy());res.append(float((e.cv(u)-vals[k]*u).norm())/max(abs(vals[k]),1e-30))
  rules=read(ROOT/'reports/gn_functional_geometry_v1/PROTOCOL.json')['numerics'];assert orth<=rules['orthogonal_max'] and max(res)<=rules['eigen_relative_residual_max'] and min(vals)>=-1e-10*max(vals)
  q=np.stack(q,1);engine.npz(destination/'FINGERPRINTS.npz',q=q,latent=latent,eigenvalues=vals)
  meta=dict(rank=32,orthogonal_error=orth,max_eigen_residual=max(res),jvps=e.jvps,vjps=e.vjps,cv_calls=e.cvs,seconds=time.perf_counter()-begin,latent_seconds=latent_seconds,action_metric_sha256=sha(ROOT/'reports/gn_functional_geometry_v1/ACTION_METRIC.json'),basis_scope='mean-action policy-head GN at this source state; full reference pool; no probe access')
  del e,head,U;gc.collect();torch.cuda.empty_cache();return q,latent,meta
 finally:g.restore(algo,before);assert all(g.compare_states(before,g.capture(algo)).values())

def generate(K,f,un,initial,destination,context):
 """Same formal solver on subproblems; full normalized KKT after every call."""
 m=len(f);Ks=(K+K.T)/2;valid=np.flatnonzero(np.diag(Ks)>1e-12);scale=np.sqrt(np.diag(Ks)[valid]);A=Ks[np.ix_(valid,valid)]/scale[:,None]/scale[None,:];b=f[valid]/scale/max(un,1e-30)
 C=sorted(initial);trace=[];total=time.perf_counter();lam=np.zeros(m);sol=None;iterations=0;fallbacks=0
 for iteration in range(m):
  engine.npz(destination/f'round{iteration:03}'/'INPUT.npz',K=Ks[np.ix_(C,C)],f=f[C],u_norm=np.array(un),original_indices=np.array(C))
  begin=time.perf_counter();sol=solve(Ks[np.ix_(C,C)],f[C],un,diag_dir=destination/f'round{iteration:03}',context=context);seconds=time.perf_counter()-begin
  lam[:]=0;lam[C]=sol['lam'];x=lam[valid]*scale/max(un,1e-30);r=b-A@x;parts=dual_kkt(A,b,x);violated=valid[r>1e-8].tolist();omitted=[int(i) for i in violated if i not in C]
  iterations+=sol['iterations'];fallbacks+=int(sol.get('fallback',{}).get('triggered',False))+int(sol.get('second_fallback',{}).get('triggered',False))
  row=dict(round=iteration,C=C.copy(),k=len(C),full_kkt=max(parts),primal_positive=parts[0],nonnegative_violation=parts[1],complementarity=parts[2],full_violations=violated,omitted_violations=omitted,max_full_violation=float(np.maximum(r,0).max(initial=0)),solve_seconds=seconds,solver_iterations=sol['iterations'],fallbacks=fallbacks)
  trace.append(row);emit(destination/'TRACE.jsonl',row)
  if max(parts)<=1e-8:return lam,dict(status='PASS',final_size=len(C),added=len(C)-len(initial),resolves=iteration,rounds=iteration+1,kkt=max(parts),iterations=iterations,fallbacks=fallbacks,solver_seconds=sum(t['solve_seconds'] for t in trace),cg_seconds=time.perf_counter()-total),trace
  if not omitted:raise RuntimeError('FULL_KKT_UNRESOLVED_WITH_NO_OMITTED_VIOLATION')
  lookup={int(v):float(r[j]) for j,v in enumerate(valid)};add=min(omitted,key=lambda j:(-lookup[j],j));row['added_next']=add;C=sorted(C+[add])
 raise RuntimeError('CONSTRAINT_GENERATION_BUDGET_EXHAUSTED')

def correction_from_lambda(H,lam,u):
 l=torch.as_tensor(lam,device=H.device,dtype=torch.float64);c=torch.empty_like(u)
 for j in range(0,len(u),65536):c[j:j+65536]=(H[j:j+65536].double()@l).float()
 return c

def probe_branch(algo,ds,base,rows,d,fullrows,role):
 data=native.probe_data(algo,ds,rows);native.apply(algo,base,d,1.);algo.policy.eval();out=[]
 try:
  with torch.no_grad():
   for row,datum in zip(rows,data):
    previous=fullrows.loc[row['record_id']];loss=float(native.single_loss(algo,datum));out.append(dict(record_id=row['record_id'],task_id=row['task_id'],trajectory_id=row['trajectory_id'],role=role,base_loss=float(previous.base_loss),tau_zero=float(previous.tau_zero),delta=loss-float(previous.base_loss),delta_full=float(previous.measured_delta_dfull),delta_native=float(previous.measured_delta_d0)))
 finally:g.restore(algo,base)
 return out

def run_start(algo,ds,start,refs,old,new):
 dest=OUT/'worksets'/start['start_id'];dest.mkdir(parents=True,exist_ok=True);base=g.capture(algo);records=[];selections={};allprobes=[]
 try:
  d0,dfull,full,H,K,w,proposal=engine.corrected(algo,ds,start,refs,base,dest/'full_recheck')
  if dfull is None:raise RuntimeError('FULL_REPRODUCTION_SOLVER_FAILED')
  original=read(OUT/'starts'/start['start_id']/'COMPLETE.json')
  for key in ['native_norm','corrected_norm','kkt_after','correction_ratio']:assert np.isclose(full[key],original[key],rtol=1e-12,atol=1e-12),(key,full[key],original[key])
  snapshots=w.snapshot_parameters();g.restore(algo,proposal['candidate']);u=w.displacement(snapshots);g.restore(algo,base);_,f=inner(H,u);un=float(u.double().norm());m=len(refs);k=m//2;order=canonical(refs)
  fo=pd.read_csv(OUT/'starts'/start['start_id']/'OLD.csv').set_index('record_id');fn=pd.read_csv(OUT/'starts'/start['start_id']/'NEW.csv').set_index('record_id')
  q=latent=None;geometry_error=None
  try:q,latent,gm=fingerprint(algo,ds,refs,dest);write(dest/'GN_CONTEXT.json',gm)
  except Exception:geometry_error=traceback.format_exc();write(dest/'GN_FAILURE.json',dict(error=geometry_error));g.restore(algo,base)
  if latent is None:
   begin=time.perf_counter();algo.policy.eval()
   with torch.no_grad():latent=np.concatenate([b['feature'].squeeze(1).mean(tuple(range(1,b['feature'].squeeze(1).ndim-1))).cpu().numpy() for b in engine.refs_batches(algo,ds,refs)])
   gm=dict(latent_seconds=time.perf_counter()-begin);g.restore(algo,base)
  methods=['RANDOM0','RANDOM1','RANDOM2','STRIDE','LATENT','GN']
  fullold=float(fo.measured_delta_dfull.mean());fullnew=float(fn.measured_delta_dfull.mean())
  fullrec=dict(start_id=start['start_id'],seed=start['seed'],transition=start['transition'],method='FULL',status='PASS',initial_size=m,final_size=m,added=0,resolves=0,kkt=full['kkt_after'],iterations=full['iterations'],total_seconds=full['projection_total_seconds'],selection_seconds=0,solver_seconds=full['solve_seconds'],update_relative_error=0,update_cosine=1,old_gap=0,new_gap=0,harm_fraction_gap=0,peak_gpu_bytes=torch.cuda.max_memory_allocated())
  records.append(fullrec)
  for method in methods:
   md=dest/method;md.mkdir(exist_ok=True);begin=time.perf_counter();rec=dict(start_id=start['start_id'],seed=start['seed'],transition=start['transition'],method=method,initial_size=k)
   try:
    if method=='GN' and q is None:raise RuntimeError('GN_BASIS_NOT_CERTIFIED: '+str(geometry_error))
    if method.startswith('RANDOM'):chosen=sorted(np.random.RandomState(int(method[-1])).choice(m,k,replace=False).tolist());feature_seconds=0.
    elif method=='STRIDE':chosen=stride(refs,k);feature_seconds=0.
    else:chosen=fps(q if method=='GN' else latent,k,order);feature_seconds=gm['seconds'] if method=='GN' else gm['latent_seconds']
    selection_seconds=time.perf_counter()-begin+feature_seconds;selections[method]=chosen
    lam,diagnostic,trace=generate(K,f,un,chosen,md,dict(start_id=start['start_id'],method=method,formal_training_resumed=False));c=correction_from_lambda(H,lam,u)
    _,predicted=inner(H,u-c);hn=np.sqrt(np.maximum(np.diag(K),0));valid=np.flatnonzero(np.diag(K)>1e-12);physical=float(np.maximum(predicted[valid]/np.maximum(hn[valid]*un,1e-30),0).max(initial=0));assert physical<=5e-6
    g.restore(algo,proposal['candidate']);w.subtract_correction(c);actual=w.displacement(snapshots);_,rates=inner(H,actual);an=float(actual.double().norm());assert int((rates>5e-6*hn*an+1e-15).sum())==0
    d=native.flat([p.detach().double()-base['model'][n].to(p.device).double() for n,p in algo.policy.named_parameters() if p.requires_grad]);error=float((d-dfull).norm()/max(float(dfull.norm()),1e-30));cos=float(torch.dot(d,dfull)/(d.norm()*dfull.norm())) if d.norm()*dfull.norm()>0 else None
    algorithm_seconds=time.perf_counter()-begin+feature_seconds;g.restore(algo,base)
    prows=probe_branch(algo,ds,base,old,d,fo,'old')+probe_branch(algo,ds,base,new,d,fn,'new')
    for row in prows:row.update(start_id=start['start_id'],seed=start['seed'],transition=start['transition'],method=method)
    allprobes.extend(prows);od=pd.DataFrame(prows[:len(old)]);nd=pd.DataFrame(prows[len(old):]);initialA=K[np.ix_(chosen,chosen)];sc=np.sqrt(np.maximum(np.diag(initialA),1e-300));initialA=initialA/sc[:,None]/sc[None,:];off=initialA.copy();np.fill_diagonal(off,0)
    rec.update(**diagnostic,selection_seconds=selection_seconds,total_seconds=algorithm_seconds,update_relative_error=error,update_cosine=cos,physical_residual=physical,old_gap=float((od.delta-od.delta_full).mean()),new_gap=float((nd.delta-nd.delta_full).mean()),harm_fraction_gap=float((od.delta>od.tau_zero).mean()-(od.delta_full>od.tau_zero).mean()),introduced=int(((od.delta_native<=od.tau_zero)&(od.delta>od.tau_zero)).sum()),prevented=int(((od.delta_native>od.tau_zero)&(od.delta<=od.tau_zero)).sum()),gram_condition=float(np.linalg.cond(initialA)),numerical_rank=int(np.linalg.matrix_rank(initialA)),max_correlation=float(abs(off).max()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),first_scan_violation=trace[0]['max_full_violation'],first_scan_violated=len(trace[0]['full_violations']))
    table(md/'PROBES.csv',prows);del d,c
   except Exception:rec.update(status='FAILED',error=traceback.format_exc(),total_seconds=time.perf_counter()-begin)
   finally:g.restore(algo,base)
   records.append(clean(rec));write(md/'RESULT.json',clean(rec))
  write(dest/'SELECTIONS.json',selections);table(dest/'PROBES.csv',allprobes);table(dest/'METHODS.csv',records)
  assert all(g.compare_states(base,g.capture(algo)).values());write(dest/'COMPLETE.json',dict(start_id=start['start_id'],methods=records,formal_training_resumed=False))
 finally:g.restore(algo,base)

def summarize_worksets():
 starts=read(OUT/'START_MANIFEST.json')['starts'];rows=[];probes=[];selections={};traces=[]
 for s in starts:
  p=OUT/'worksets'/s['start_id'];assert (p/'COMPLETE.json').exists();rows.extend(read(p/'COMPLETE.json')['methods']);probes.append(pd.read_csv(p/'PROBES.csv'));selections[s['start_id']]=read(p/'SELECTIONS.json')
  for tr in p.glob('*/TRACE.jsonl'):
   for line in tr.read_text().splitlines():traces.append(dict(start_id=s['start_id'],method=tr.parent.name,**__import__('json').loads(line)))
 d=pd.DataFrame(rows);d.to_csv(OUT/'WORKSET_SOLVER_DIAGNOSTICS.csv',index=False);d.to_csv(OUT/'WORKSET_UPDATE_COMPARISON.csv',index=False);pd.concat(probes).to_csv(OUT/'WORKSET_PROBE_RESULTS.csv',index=False);table(OUT/'CONSTRAINT_GENERATION_TRACE.csv',traces);table(OUT/'FULL_CONSTRAINT_SCAN.csv',[r for r in traces if r['round']==0]);write(OUT/'WORKSET_SELECTIONS.json',selections)
 evidence=[]
 for cell,c in d.groupby(['seed','transition']):
  gn=c[c.method=='GN'].set_index('start_id');comparisons=[]
  for method in ['STRIDE','LATENT']:
   other=c[c.method==method].set_index('start_id');pair=gn.join(other,lsuffix='_gn',rsuffix='_other');ok=(pair.status_gn=='PASS')&(pair.status_other=='PASS');v=pair[ok]
   wins=dict(smaller_workset=bool(len(v)>0 and (v.final_size_gn-v.final_size_other).median()<0),less_failure=bool((gn.status!='PASS').sum()<(other.status!='PASS').sum()),faster_total=bool(len(v)>0 and (v.total_seconds_gn/v.total_seconds_other).median()<.95),closer_update=bool(len(v)>0 and (v.update_relative_error_gn-v.update_relative_error_other).median() < -1e-5),better_protection=False)
   # NLL differences smaller than the original per-start deadband do not count as advantages.
   sm=pd.read_csv(OUT/'START_LEVEL_METRICS.csv').set_index('start_id');tau=sm.loc[v.index,'tau'];wins['better_protection']=bool(len(v)>0 and ((v.old_gap_gn-v.old_gap_other)/tau).median() < -1)
   no_cost=bool(len(v)==20 and ((v.new_gap_gn-v.new_gap_other)/sm.loc[v.index,'new_tau']).median()<=1)
   comparisons.append(dict(comparator=method,wins=wins,pass_cell=bool(sum(wins.values())>=2 and no_cost and len(v)==20),new_cost_within_deadband=no_cost,paired_starts=len(v)))
  evidence.append(dict(seed=int(cell[0]),transition=cell[1],comparisons=comparisons,passed=all(x['pass_cell'] for x in comparisons)))
 failed=int((d.status!='PASS').sum());support=failed==0 and sum(c['passed'] for c in evidence)>=5
 label='SUPPORTED' if support else ('INCONCLUSIVE' if failed else 'NOT_SUPPORTED')
 write(OUT/'GN_WORKSET_VALUE_REPORT.json',clean(dict(classification=label,cells=evidence,failed_methods=failed,GN_cost_included=True,not_training_speedup=True,random_replicates='0/1/2 all reported; never select best',method_summaries={method:{k:stats(v[k]) for k in ['final_size','total_seconds','update_relative_error','old_gap','new_gap']} for method,v in d.groupby('method')},conclusion='GN practical advantage established on tested local states only' if support else 'GN redundancy signal has no established practical advantage over cheap input-space compression.')))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for name,key in [('workset_method_vs_solver_time','total_seconds'),('workset_method_vs_update_error','update_relative_error'),('workset_method_vs_old_protection','old_gap')]:
  fig,ax=plt.subplots();groups=list(d.groupby('method'));ax.boxplot([v[key].dropna() for _,v in groups],labels=[m for m,_ in groups]);ax.set_ylabel(key);ax.tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=130);plt.close(fig)
 fig,ax=plt.subplots();ax.scatter(d.final_size,d.first_scan_violation,s=8);ax.set(xlabel='Final candidate set size',ylabel='Initial full-set violation');fig.tight_layout();fig.savefig(OUT/'workset_size_vs_full_violation.png',dpi=130);plt.close(fig)
