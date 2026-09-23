import os,time,gc,platform,subprocess,traceback,resource,hashlib
from pathlib import Path
import numpy as np,torch,pandas as pd
from gradient_relation_experiment import kernel as g
from actual_update_fidelity_v1 import engine as n
from qp_corrected_update_fidelity_v1 import engine as e
from qp_corrected_update_fidelity_v1.run import build
from sample_qp_optimized_v1_continuation_v4.core import Workspace,inner,project,Timings
from qp_writeback_forensic_v1.run import scatter
from .common import *
from .materialize import apply,bins

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rss_bytes():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
def sync():
 if torch.cuda.is_available():torch.cuda.synchronize()

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 assert 'OK' in (OUT/'tests.log').read_text()
 write(OUT/'UNIT_TEST_REPORT.json',dict(status='PASS',tests=6,log_sha256=sha(OUT/'tests.log'),includes=['toy actual parameter rounding constraint flip','W1 exact reread','model optimizer scheduler RNG restore','residual decomposition identity','W1 W2 bitwise repetition','ULP bucket endpoints']))
 context=dict(branch=subprocess.check_output(['git','branch','--show-current']).decode().strip(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),status=subprocess.check_output(['git','status','--short','--untracked-files=no']).decode(),python=platform.python_version(),pytorch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']).decode().strip(),solver='unchanged sample_qp_optimized_v1_continuation_v4',kkt=1e-8,physical_threshold=5e-6,formal_training_resumed=False)
 write(OUT/'EXPERIMENT_CONTEXT.json',context)
 protocol=dict(targets=TARGETS,threshold=5e-6,kkt=1e-8,denominator='fixed norm of exact native float32 candidate minus source, evaluated in float64; also record frozen formal solver u32 norm',dstar='unchanged forensic ideal d64 = promote(formal float32 u)-H64 lambda64; original QP untouched',W0='physical native parameters then formal float32 correction subtraction',W1='restore theta_before; build full float64(theta_before)+dstar64; one final fp32 write',W2='restore theta_before; build float64(theta_before)+float64(formal dstar32); one final fp32 write',old_state_mutation='candidate proposal is generated transactionally once; materialization branches never step optimizer or scheduler and restore all states',ulp_bins=['<0.5','0.5-1','1-2','2-4','>=4'],top_contributions='top20 by absolute coordinate contribution; report absolute share, signed/net ratio and positive share separately; fractions of coordinates are not energy fractions',near_active='abs(ideal dot)/(h_norm*d0_norm)<=1e-8; active normalized dual>1e-12',rounding_bound='requested +inf spacing at nearest float32 ideal target, plus maximum up/down neighbor gap bound to cover exponent-boundary asymmetry',stage_A_gate='both W1 physical residual <=5e-6',stage_A_failure='FP32_GRID_FEASIBILITY_NOT_RESOLVED; stop, no120 regression, no repair runs',stage_B='only A passes; same120 starts, all original QP/ref/probe/gates; W1 only; any physical failure blocks supported conclusion',stage_C='only120pass AND previous protection/fidelity gates pass; same corrected candidates, evaluation-only buffer trajectories excluded from references and replay; unavailable explicitly retained',GN_workset=False,formal_training_resumed=False)
 write(OUT/'PROTOCOL.json',protocol)
 old=read(SOURCE/'EXECUTION_FREEZE.json');assert all(sha(ROOT/p)==v for p,v in old['files'].items())
 files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'qp_fp32_writeback_validation_v1').glob('*.py')};files.update(old['files'])
 for sid in TARGETS:
  for path in [FORENSIC/sid/'RESULT.json',FORENSIC/sid/'CONSTRAINT_RESIDUALS.npz',FORENSIC/sid/'ROUNDING_COORDINATES.npz',SOURCE/'starts'/sid/'QP_INPUT.npz']:files[str(path.relative_to(ROOT))]=sha(path)
 write(OUT/'FREEZE.json',dict(files=files,starting_head=context['head']))

def measure(s):
 sid=s['start_id'];out=OUT/sid;out.mkdir(exist_ok=True);state_path=Path(f'/root/actual_update_fidelity_v1/rolling_s45_t{s["task_id"]}.pt');source_sha=sha(state_path)
 saved=torch.load(state_path,map_location='cpu');assert saved['position']==s['stage_step'];assert saved['manifest_sha256']==sha(ROOT/'reports/actual_update_fidelity_v1/START_MANIFEST.json')
 assert all(g.digest(saved['state'][key])==h for key,h in s['expected_witness'].items())
 algo,ds,_=build(s);g.restore(algo,saved['state']);base=g.capture(algo);restore=g.compare_states(saved['state'],base);assert all(restore.values());del saved
 refs=read(SOURCE/'REFERENCE_MANIFEST.json')['cells'][f'{s["seed"]}/{s["task_id"]}'];comparison=[];decomposition=[];contributions=[];margins=[];ulp_report={};counts=[];times={};actual_checks={};begin=time.perf_counter()
 try:
  proposal=n.proposal(algo,g.batch(ds,s['record_indices']),base);w=Workspace(n.ps(algo));before=w.snapshot_parameters();H,K=e.reference_matrix(algo,ds,refs,w);g.restore(algo,proposal['candidate']);u=w.displacement(before);_,f=inner(H,u);un=float(u.double().norm());d0=proposal['d0'];dn=float(d0.norm())
  previous=np.load(SOURCE/'starts'/sid/'QP_INPUT.npz');same={key:bool(np.array_equal(value,previous[key])) for key,value in [('K',K),('f',f),('u_norm',np.array(un))]};assert all(same.values())
  corr,sol=project(H,K,u,Timings(True),diag_dir=out/'solver',context=dict(start_id=sid,validation_only=True));old=np.load(FORENSIC/sid/'CONSTRAINT_RESIDUALS.npz');assert np.array_equal(sol['lam'],old['lam']);write(out/'SOLUTION.json',sol)
  c64=torch.empty_like(u,dtype=torch.float64);lam=torch.as_tensor(sol['lam'],device=u.device,dtype=torch.float64)
  for j in range(0,len(u),65536):c64[j:j+65536]=H[j:j+65536].double()@lam
  d64=u.double()-c64;d32=u-corr;dsct=scatter(u,corr,w.parameters);assert torch.equal(dsct,d32) and torch.equal(c64.float(),corr)
  _,ideal=inner(H,d64);assert np.array_equal(ideal,old['d64_rates'])
  z=np.load(FORENSIC/sid/'ROUNDING_COORDINATES.npz');indices=torch.as_tensor(z['indices'],device=u.device);assert np.array_equal(d64[indices].cpu().numpy(),z['d64']);assert np.array_equal(d32[indices].cpu().numpy(),z['d32'])
  params=list(w.parameters);theta0=n.flat(before);native_parameters=tuple(proposal['candidate']['model'][name].to(p.device) for name,p in algo.policy.named_parameters() if p.requires_grad)
  reconstruction=n.flat([(v.double()-b.double()) for v,b in zip(native_parameters,before)]);assert torch.equal(reconstruction,d0)
  check=dict(state_id=sid,status='PASS',state_path=str(state_path),sha256=source_sha,restore=restore,global_step=s['global_step'],optimizer_step=base['optimizer_step'],scheduler_step=base['scheduler_step'],same_saved_QP=same,lambda_bitwise=True,dstar_direct_rates_bitwise=True,dstar_saved_coordinates_bitwise=True,d0_exact_native_difference=True,d0_sha256=tensor_hash(d0),dstar64_sha256=tensor_hash(d64),dstar32_sha256=tensor_hash(d32),original_formal_u32_norm=un,fixed_d0_norm=dn,reference_manifest_sha256=sha(SOURCE/'REFERENCE_MANIFEST.json'),preceding_batches_replayed=0)
  write(out/'RESTORE_CHECK.json',check)
  valid=np.array(sol['valid'],int);hn=np.sqrt(np.maximum(np.diag(K),0));den=np.maximum(hn*dn,1e-30)
  for name,d in [('d64',d64),('d32',d32),('dscatter',dsct)]:
   _,dot=inner(H,d);counts.append(dict(state_id=sid,stage=name,max_positive_residual=float(np.maximum(dot[valid]/den[valid],0).max())))
  t=time.perf_counter();ulp_before=(torch.nextafter(theta0,torch.full_like(theta0,float('inf'))).double()-theta0.double()).abs();correction=d0-d64
  for label,value in [('correction',correction),('final_displacement',d64)]:
   ratio=value.abs()/ulp_before;ulp_report[label]=dict(counts=bins(ratio),nonzero_counts=bins(ratio[value!=0]),total=len(value),nonzero=int((value!=0).sum()))
  target64=theta0.double()+d64;rounded=target64.float();up=torch.nextafter(rounded,torch.full_like(rounded,float('inf'))).double()-rounded.double();down=rounded.double()-torch.nextafter(rounded,torch.full_like(rounded,float('-inf'))).double();half_up=up*.5;half_safe=torch.maximum(up,down)*.5
  bound=torch.zeros(len(refs),device=u.device,dtype=torch.float64);safe=bound.clone()
  for j in range(0,len(u),65536):
   block=H[j:j+65536].double().abs();bound+=block.T@half_up[j:j+65536];safe+=block.T@half_safe[j:j+65536]
  bound=bound.cpu().numpy();safe=safe.cpu().numpy();x=np.zeros(len(refs));x[valid]=sol['normalized_x'];near=np.abs(ideal)/den<=1e-8
  old_fail=read(FORENSIC/sid/'RESULT.json')['failed_constraint_indices'];rates_by_method={}
  ulp_seconds=time.perf_counter()-t
  for method in ['W0','W1','W2']:
   g.restore(algo,base);assert all(g.compare_states(base,g.capture(algo)).values());sync();gpu0=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();cpu0=rss_bytes();t=time.perf_counter()
   dw=apply(params,before,method,d64,d32,corr,native_parameters);sync();elapsed=time.perf_counter()-t;peak_extra=torch.cuda.max_memory_allocated()-gpu0;rss_delta=rss_bytes()-cpu0
   # Confirm rerun determinism from original source, no native load in W1/W2.
   first_hash=tensor_hash(n.flat([p.detach() for p in params]));g.restore(algo,base);dw_repeat=apply(params,before,method,d64,d32,corr,native_parameters);assert torch.equal(dw,dw_repeat);del dw_repeat
   _,written=inner(H,dw);rates_by_method[method]=written;normalized=written/den;failed=np.flatnonzero(normalized>5e-6).tolist();error=dw-d64;_,edot=inner(H,error);closure=float(np.max(abs(written-ideal-edot)));assert closure<1e-13
   if method=='W0':assert np.array_equal(written,old['dwritten_rates']),('W0_NOT_REPRODUCED',written,old['dwritten_rates'])
   r=dict(state_id=sid,method=method,max_positive_residual=float(np.maximum(normalized[valid],0).max()),violating_count=len(failed),write_error_l2=float(error.norm()),write_error_linf=float(error.abs().max()),write_error_relative_to_dstar=float(error.norm()/d64.norm()),write_error_relative_to_correction=float(error.norm()/correction.norm()),physical_pass=bool(np.max(normalized[valid])<=5e-6),materialization_seconds=elapsed,incremental_gpu_peak_bytes=peak_extra,rss_delta_bytes=rss_delta,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,full_float64_target_bytes=len(u)*8 if method!='W0' else 0,parameters_sha256=first_hash,deterministic=True,closure_max=closure,denominator_d0_norm=dn)
   comparison.append(r);counts.append(dict(state_id=sid,stage=method,max_positive_residual=r['max_positive_residual']))
   error_ulps=error.abs()/ulp_before;ulp_report[method]=dict(error_abs_over_before_ulp_counts=bins(error_ulps),error_l2=r['write_error_l2'],error_linf=r['write_error_linf'],error_max_abs_in_before_ulp=float(error_ulps.max()),residual_quantiles=np.quantile(normalized,[0,.25,.5,.75,.9,.99,1]).tolist(),actual_rounding_exceeds_half_up=int((error.abs()>half_up).sum()),actual_rounding_exceeds_safe_half_gap=int((error.abs()>half_safe+torch.finfo(torch.float64).eps*target64.abs()).sum()))
   focus=sorted(set(old_fail+failed));t=time.perf_counter()
   for j,ref in enumerate(refs):
    row=dict(state_id=sid,method=method,constraint=j,origin=ref['origin'],task_id=ref['task_id'],r_ideal=float(ideal[j]),r_written=float(written[j]),delta_r=float(edot[j]),normalized=float(normalized[j]),closure=float(written[j]-ideal[j]-edot[j]),physical_failed=j in failed)
    if j in focus:
     products=H[:,j].double()*error;abs_total=float(products.abs().sum());pos_total=float(products.clamp_min(0).sum());indices=torch.topk(products.abs(),20).indices;values=products[indices];net=float(edot[j])
     for k in [1,5,10,20]:
      row[f'top{k}_absolute_share']=float(values[:k].abs().sum())/max(abs_total,1e-300);row[f'top{k}_signed_over_net']=float(values[:k].sum())/net if net!=0 else None;row[f'top{k}_positive_share']=float(values[:k].clamp_min(0).sum())/max(pos_total,1e-300)
     for rank,index in enumerate(indices.tolist()):contributions.append(dict(state_id=sid,method=method,constraint=j,rank_abs=rank+1,coordinate=index,h=float(H[index,j]),error=float(error[index]),product=float(products[index]),theta_before=float(theta0[index]),theta_target64=float(target64[index]),dstar64=float(d64[index]),ulp_before=float(ulp_before[index])))
     del products
    decomposition.append(row)
    margins.append(dict(state_id=sid,method=method,constraint=j,margin=float(-ideal[j]),requested_bound=float(bound[j]),conservative_neighbor_bound=float(safe[j]),margin_over_bound=float(-ideal[j]/max(bound[j],1e-300)),active=bool(x[j]>1e-12),near_active=bool(near[j]),margin_below_bound=bool(-ideal[j]<bound[j]),physical_failed=j in failed))
   ulp_seconds+=time.perf_counter()-t
   after=g.capture(algo);state_checks={k:g.digest(after[k])==g.digest(base[k]) for k in ['optimizer','scheduler','rng','optimizer_step','scheduler_step','training_modes']};assert all(state_checks.values());g.restore(algo,base);restored=g.compare_states(base,g.capture(algo));assert all(restored.values());actual_checks[method]=dict(nonmodel_state_unchanged=state_checks,full_restore=restored,deterministic=True)
   del dw,error
  ulp_report['diagnostic_seconds']=ulp_seconds;ulp_report['bin_labels']=read(OUT/'PROTOCOL.json')['ulp_bins'];ulp_report['note']='All-coordinate and nonzero counts separate. Top shares use absolute contribution denominator; signed/net shares may exceed100% due cancellation. Zero coordinates not excluded from all-coordinate bins.'
  e.npz(out/'QP_AND_RESIDUALS.npz',K=K,f=f,u_norm=np.array(un),exact_d0_norm=np.array(dn),A=previous['A'],b=previous['b'],lam=np.array(sol['lam']),ideal=ideal,bound=bound,safe_bound=safe,**rates_by_method)
  table(out/'MATERIALIZATION_COMPARISON.csv',comparison);table(out/'CONSTRAINT_ERROR_DECOMPOSITION.csv',decomposition);table(out/'TOP_COORDINATES.csv',contributions);table(out/'ROUNDING_MARGIN_ANALYSIS.csv',margins);table(out/'STAGES.csv',counts);write(out/'ULP_ANALYSIS.json',ulp_report);write(out/'STATE_CHECKS.json',actual_checks)
  assert sha(state_path)==source_sha
  result=dict(state_id=sid,status='COMPLETE',restore=check,comparison=comparison,seconds=time.perf_counter()-begin,formal_training_resumed=False);write(out/'COMPLETE.json',result);return result
 finally:g.restore(algo,base)

def summarize(results):
 comps=[];decomps=[];margins=[];tops=[];stages=[];ulps={};checks=[]
 for result in results:
  path=OUT/result['state_id'];comps.extend(result['comparison']);decomps.append(pd.read_csv(path/'CONSTRAINT_ERROR_DECOMPOSITION.csv'));margins.append(pd.read_csv(path/'ROUNDING_MARGIN_ANALYSIS.csv'));tops.append(pd.read_csv(path/'TOP_COORDINATES.csv'));stages.append(pd.read_csv(path/'STAGES.csv'));ulps[result['state_id']]=read(path/'ULP_ANALYSIS.json');checks.append(result['restore'])
 table(OUT/'MATERIALIZATION_COMPARISON.csv',comps);pd.concat(decomps).to_csv(OUT/'CONSTRAINT_ERROR_DECOMPOSITION.csv',index=False);pd.concat(margins).to_csv(OUT/'ROUNDING_MARGIN_ANALYSIS.csv',index=False);pd.concat(tops).to_csv(OUT/'TOP_COORDINATES.csv',index=False);write(OUT/'ULP_ANALYSIS.json',ulps);write(OUT/'FAILURE_STATE_RESTORE_CHECK.json',dict(status='PASS',states=checks))
 w1=[r for r in comps if r['method']=='W1'];w2=[r for r in comps if r['method']=='W2'];passed=all(r['physical_pass'] for r in w1);p2=all(r['physical_pass'] for r in w2)
 root='DOUBLE_MATERIALIZATION' if passed and p2 else ('MIXED' if passed or any(r['physical_pass'] for r in w1) else 'FP32_GRID_LIMIT')
 summary=dict(status='STAGE_A_COMPLETE',FP32_WRITEBACK_ROOT_CAUSE=root,SINGLE_TARGET_MATERIALIZATION='SUPPORTED' if passed else 'NOT_SUPPORTED',stage_A_decision='SINGLE_TARGET_MATERIALIZATION_SUPPORTED' if passed else 'FP32_GRID_FEASIBILITY_NOT_RESOLVED',WRITEBACK_120='NOT_RUN',QP_PROTECTION='NOT_REEVALUATED',POST_QP_FIRST_ORDER_FIDELITY='NOT_REEVALUATED',PLASTICITY_COST='NOT_REEVALUATED',TRAJECTORY_DISJOINT_PROTECTION='NOT_RUN',NEXT_STAGE='MORE_VALIDATION_REQUIRED' if passed else 'PHYSICAL_REPAIR_REQUIRED',stage_B_allowed=passed,formal_training_resumed=False,GN_workset_executed=False)
 write(OUT/'SUMMARY.json',summary)
 report=['# FP32 QP physical-writeback validation v1','','Stage A uses only the two frozen complete source states; no preceding batches replayed. Solver/QP/formal runtime and prior results remain unchanged.','', '|state|method|max positive normalized residual|violations|physical pass|write error L2|','|---|---|---:|---:|---|---:|']
 for r in comps:report.append(f"|{r['state_id']}|{r['method']}|{r['max_positive_residual']:.12g}|{r['violating_count']}|{r['physical_pass']}|{r['write_error_l2']:.12g}|")
 report+=['','```json',__import__('json').dumps(summary,indent=2),'```','','## Answers and limits','1. Source state, native candidate, same K/f/lambda, dstar rates and archived coordinates are verified against forensic artifacts. W0 exact readback residuals reproduce bitwise.',f'2–4. W1 two-state gate={passed}; W2 two-state gate={p2}. Root classification={root}. A grid-limit label means nearest-FP32 materialization of these fixed targets failed; it does not claim no FP32 feasible update exists.', '5–7. ULP bins, signed/absolute top-coordinate shares, ideal margins and streaming rounding bounds are retained in ULP_ANALYSIS and the three coordinate/constraint CSVs. The requested upward-ULP bound and a maximum-neighbor-gap bound are both reported, because binade boundaries can have asymmetric spacing.', '8–13. StageB/C are not run unless preceding gates pass. No120-state or new probe conclusion is fabricated, and prior118-state results are unchanged.', '14. Actual materialization wall time, GPU peak increment, current RSS delta, process peak RSS, and full double target bytes are recorded per branch. StageA resource data do not demonstrate end-to-end training cost.', '15. The continuous solver has already certified the original QP. This experiment provides no evidence requiring a QP objective/constraint change: solver/QP semantics do not need modification based on this diagnosis. Physical feasibility handling remains unresolved if W1 fails.', '16. GN workset is prohibited in this experiment; a failed StageA gate does not authorize it.']
 if not passed:report+=['','## Proposals only; not executed','- P1 physical line search: test the preregisterable fixed alpha sequence 1,63/64,31/32,15/16,7/8,3/4,1/2,0 and choose the largest actually feasible materialized update. Pros: simple, full scan. Risks: quantized feasibility is not monotonic in alpha, plasticity may shrink; zero is a fallback that gives no learning.', '- P2 materialize/check/corrective repair: use violated/near-active coordinates to propose a small correction, then recheck every original constraint after materialization. Pros: directly targets observed violation. Risks: further rounding, cycling and budget exhaustion; final full scan is essential.', '- P3 quantization-aware feasibility: include Q_theta(d)=FP32(theta+d)-theta in the feasible-update definition. Pros: matches the executed object. Risks: discrete/non-smooth problem and higher implementation/validation cost.', 'None of these proposals was implemented or tested. No negative margin, threshold change or extra solver was used.']
 (OUT/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n');plots(pd.concat(stages),pd.concat(decomps),pd.concat(margins),ulps)
 return summary

def plots(stages,decomp,margins,ulps):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots()
 for sid,d in stages.groupby('state_id'):ax.plot(d.stage,d.max_positive_residual,'o-',label=sid)
 ax.axhline(5e-6,color='red',linestyle='--',label='fixed physical gate');ax.set(yscale='log',ylabel='Positive residual / (||h|| ||d0||)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'materialization_residuals.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,(sid,u) in zip(axes,ulps.items()):
  for k in ['correction','final_displacement']:ax.plot(u['bin_labels'],np.array(u[k]['counts'])/u[k]['total'],'o-',label=k)
  ax.set(title=sid,ylabel='Fraction of all parameter coordinates');ax.legend(fontsize=7);ax.tick_params(axis='x',rotation=30)
 fig.tight_layout();fig.savefig(OUT/'correction_vs_ulp_histogram.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots()
 for method,d in decomp.groupby('method'):
  v=d[d.top20_absolute_share.notna()];ax.scatter(v.constraint,v.top20_absolute_share,label=method,s=15)
 ax.set(xlabel='Constraint index',ylabel='Top20 absolute-contribution share');ax.legend();fig.tight_layout();fig.savefig(OUT/'write_error_constraint_contributions.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();d=margins[margins.method=='W1'];ax.scatter(d.requested_bound,d.margin,c=d.physical_failed.astype(int));ax.set(xlabel='Sum |h| half target ULP bound',ylabel='Ideal margin (-h dot dstar)');fig.tight_layout();fig.savefig(OUT/'ideal_margin_vs_rounding_bound.png',dpi=150);plt.close(fig)

def main():
 freeze();torch.set_num_threads(2);torch.set_num_interop_threads(1);g.init_ddp(Path('/root/qp_corrected_update_fidelity_v1')/f'fp32_validation_{os.getpid()}');starts=read(SOURCE/'START_MANIFEST.json')['starts'];results=[]
 try:
  for sid in TARGETS:
   write(OUT/'STATUS.json',dict(stage='STAGE_A',state_id=sid,pid=os.getpid(),formal_training_resumed=False));results.append(measure(next(s for s in starts if s['start_id']==sid)));gc.collect();torch.cuda.empty_cache()
  summary=summarize(results);f=read(OUT/'FREEZE.json');assert all(sha(ROOT/p)==v for p,v in f['files'].items());write(OUT/'STATUS.json',dict(stage='STAGE_A_PASS_REQUIRES_B' if summary['stage_B_allowed'] else 'COMPLETE_STOP_AFTER_STAGE_A',formal_training_resumed=False))
 except Exception:
  write(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False));raise
 finally:
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()
