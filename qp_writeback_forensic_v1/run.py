import os,time,json,gc,traceback,subprocess
from pathlib import Path
import numpy as np,torch
from qp_corrected_update_fidelity_v1 import engine as e
from qp_corrected_update_fidelity_v1.run import build
from qp_corrected_update_fidelity_v1.common import ROOT,read,write,sha,table
from sample_qp_optimized_v1_continuation_v4.core import Workspace,inner,project,Timings
from actual_update_fidelity_v1 import engine as n
from gradient_relation_experiment import kernel as g
SOURCE=ROOT/'reports/qp_corrected_update_fidelity_v1'
OUT=ROOT/'reports/qp_writeback_forensic_v1'
TARGETS=['s45_t7_e98_b195','s45_t8_e99_b94']

def scatter(u,c,parameters):
 chunks=[];offset=0
 for p in parameters:
  length=p.numel();uc=u[offset:offset+length].reshape(p.shape);cc=c[offset:offset+length].reshape(p.shape)
  chunks.append((uc-cc).reshape(-1));offset+=length
 assert offset==len(u)
 return torch.cat(chunks)

def inspect_start(s):
 sid=s['start_id'];out=OUT/sid;out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
 statepath=Path(f'/root/actual_update_fidelity_v1/rolling_s45_t{s["task_id"]}.pt');sourcehash=sha(statepath)
 saved=torch.load(statepath,map_location='cpu');assert saved['position']==s['stage_step'];assert saved['manifest_sha256']==sha(ROOT/'reports/actual_update_fidelity_v1/START_MANIFEST.json')
 expected=s['expected_witness'];assert all(g.digest(saved['state'][k])==v for k,v in expected.items())
 algo,ds,_=build(s);g.restore(algo,saved['state']);base=g.capture(algo);checks=g.compare_states(saved['state'],base);assert all(checks.values());del saved
 write(out/'RESTORE.json',dict(source=str(statepath),sha256=sourcehash,position=s['stage_step'],witness=expected,checks=checks,preceding_batches_replayed=0))
 try:
  refs=read(SOURCE/'REFERENCE_MANIFEST.json')['cells'][f'{s["seed"]}/{s["task_id"]}'];proposal=n.proposal(algo,g.batch(ds,s['record_indices']),base);w=Workspace(n.ps(algo));before=w.snapshot_parameters();H,K=e.reference_matrix(algo,ds,refs,w)
  g.restore(algo,proposal['candidate']);u=w.displacement(before);_,f=inner(H,u);un=float(u.double().norm());old=np.load(SOURCE/'starts'/sid/'QP_INPUT.npz')
  same={key:bool(np.array_equal(val,old[key])) for key,val in [('K',K),('f',f),('u_norm',np.asarray(un))]};assert all(same.values()),same
  corr,sol=project(H,K,u,Timings(True),diag_dir=out/'solver',context=dict(start_id=sid,diagnostic_only=True));write(out/'SOLUTION.json',sol)
  lam=torch.as_tensor(sol['lam'],device=H.device,dtype=torch.float64);c64=torch.empty(len(u),device=u.device,dtype=torch.float64)
  for start in range(0,len(u),65536):c64[start:start+65536]=H[start:start+65536].double()@lam
  assert torch.equal(c64.float(),corr)
  d64=u.double()-c64;d32=u-corr;dsct=scatter(u,corr,w.parameters);assert torch.equal(dsct,d32)
  # This is the only physical correction write. Native AdamW was proposed once.
  w.subtract_correction(corr);dw32=w.displacement(before)
  dw64=n.flat([p.detach().double()-oldp.double() for p,oldp in zip(w.parameters,before)])
  after=g.capture(algo);preserved={k:g.digest(after[k])==g.digest(proposal['candidate'][k]) for k in ['optimizer','scheduler','rng','optimizer_step','scheduler_step','training_modes']};assert all(preserved.values())
  vectors={'d64':d64,'d32':d32,'dscatter':dsct,'dwritten':dw64,'dwritten_float32_report':dw32}
  hn=np.sqrt(np.maximum(np.diag(K),0));valid=np.array(sol['valid'],int);stages=[];constraints=[];rates={};previous=None;all_arrays={}
  for name,d in vectors.items():
   _,dot=inner(H,d);rates[name]=dot;dn=float(d.double().norm());fixed=dot/np.maximum(hn*un,1e-30);own=dot/np.maximum(hn*dn,1e-30);threshold=5e-6*hn*dn+1e-15
   rows=dict(start_id=sid,stage=name,dtype=str(d.dtype),norm=dn,raw_positive_count=int((dot[valid]>0).sum()),raw_max=float(dot[valid].max()),fixed_u_normalized_max=float(np.maximum(fixed[valid],0).max()),own_norm_normalized_max=float(np.maximum(own[valid],0).max()),prewrite_gate_violations=int((fixed[valid]>5e-6).sum()),postwrite_gate_violations=int((dot>threshold).sum()),fixed_feasibility_1e8_count=int((fixed[valid]>1e-8).sum()),worst_constraint=int(valid[np.argmax(fixed[valid])]),delta_from_previous_norm=float((d.double()-previous.double()).norm()) if previous is not None else 0.,delta_from_d64_norm=float((d.double()-d64).norm()),u_norm=un)
   stages.append(rows);previous=d
   for j,r in enumerate(refs):constraints.append(dict(stage=name,constraint=j,task=r['task_id'],origin=r['origin'],raw_dot=float(dot[j]),h_norm=float(hn[j]),fixed_u_normalized=float(fixed[j]),own_normalized=float(own[j]),postwrite_threshold=float(threshold[j]),postwrite_excess=float(dot[j]-threshold[j]),postwrite_failed=bool(dot[j]>threshold[j])))
   all_arrays[name+'_rates']=dot
  historical=read(SOURCE/'starts'/sid/'COMPLETE.json');rep=stages[-1];assert rep['postwrite_gate_violations']==historical['actual_positive'];assert np.isclose(rep['own_norm_normalized_max'],historical['physical_rate_max'],rtol=1e-12,atol=1e-15)
  # Matrix-space certification vs direct H dot d64, before any parameter casting.
  matrix=f-K@np.asarray(sol['lam']);direct=rates['d64'];matrix_check=float(np.max(abs(matrix-direct)))
  # Decompose the error into correction rounding, subtraction rounding,
  # original native-displacement subtraction, parameter write rounding, report rounding.
  original_exact=proposal['d0'];native_rounding=original_exact-u.double()
  correction_rounding=c64-corr.double();subtract_rounding=d32.double()-(u.double()-corr.double())
  write_rounding=dw64-(original_exact-corr.double());report_rounding=dw32.double()-dw64
  decomposed=d64+correction_rounding+subtract_rounding+native_rounding+write_rounding-subtract_rounding
  assert torch.allclose(decomposed,dw64,rtol=0,atol=1e-16)
  errors={'correction_cast':correction_rounding,'displacement_subtract':subtract_rounding,'native_delta_subtract':native_rounding,'parameter_write':write_rounding,'reported_delta_subtract':report_rounding}
  error_report={};error_rates={}
  for name,d in errors.items():
   _,z=inner(H,d);error_rates[name]=z;error_report[name]=dict(norm=float(d.norm()),max_abs=float(d.abs().max()),nonzero=int((d!=0).sum()),max_normalized_constraint_contribution=float(np.max(abs(z)/np.maximum(hn*un,1e-30))))
  failed=np.flatnonzero(rates['dwritten_float32_report']>5e-6*hn*float(dw32.double().norm())+1e-15).tolist();blocks=[];offset=0
  actual_params=[(name,p) for name,p in algo.policy.named_parameters() if p.requires_grad]
  for (name,p),oldp in zip(actual_params,before):
   end=offset+p.numel();err=write_rounding[offset:end];Hblock=H[offset:end][:,failed].double();contrib=(Hblock.T@err).cpu().numpy();oldtheta=oldp.detach().flatten();native_theta=proposal['candidate']['model'][name].to(p.device).flatten();written_theta=p.detach().flatten();cc=corr[offset:end]
   absorbed=(cc!=0)&(written_theta==native_theta);ulp=(torch.nextafter(native_theta,torch.full_like(native_theta,float('inf')))-native_theta).abs().double()
   row=dict(parameter=name,elements=p.numel(),write_error_norm=float(err.norm()),write_error_max=float(err.abs().max()),correction_nonzero=int((cc!=0).sum()),correction_absorbed=int(absorbed.sum()),corrected_parameter_changed=int((written_theta!=native_theta).sum()),max_write_error_in_ulp=float((err.abs()/ulp.clamp_min(1e-45)).max()),fixed_slice_start=offset,fixed_slice_end=end)
   for j,z in zip(failed,contrib):row[f'constraint_{j}_write_error_dot']=float(z)
   blocks.append(row);offset=end
  # Compact coordinate evidence, enough to inspect rounding without storing full H.
  indices=torch.topk(write_rounding.abs(),min(256,len(u))).indices
  theta0=n.flat(before);theta1=n.flat([proposal['candidate']['model'][name].to(p.device) for name,p in actual_params]);thetaw=n.flat([p.detach() for name,p in actual_params])
  e.npz(out/'ROUNDING_COORDINATES.npz',indices=indices.cpu().numpy(),theta0=theta0[indices].cpu().numpy(),theta_native=theta1[indices].cpu().numpy(),theta_written=thetaw[indices].cpu().numpy(),c64=c64[indices].cpu().numpy(),c32=corr[indices].cpu().numpy(),d64=d64[indices].cpu().numpy(),d32=d32[indices].cpu().numpy(),dwritten=dw64[indices].cpu().numpy(),write_error=write_rounding[indices].cpu().numpy())
  e.npz(out/'CONSTRAINT_RESIDUALS.npz',K=K,f=f,lam=np.asarray(sol['lam']),u_norm=np.array(un),**all_arrays,**{k+'_error_rates':v for k,v in error_rates.items()})
  table(out/'STAGES.csv',stages);table(out/'CONSTRAINTS.csv',constraints);table(out/'PARAMETER_BLOCKS.csv',blocks);write(out/'ERROR_DECOMPOSITION.json',error_report)
  first_fixed=next((v['stage'] for v in stages[:4] if v['prewrite_gate_violations']>0),None);first_own=next((v['stage'] for v in stages[:4] if v['postwrite_gate_violations']>0),None)
  result=dict(start_id=sid,status='COMPLETE',source_QP_bitwise=same,kkt=sol['kkt_error'],formal_project_residual=sol['written_normalized_rate_max'],historical_failure_exactly_reproduced=True,first_violation_fixed_native_denominator=first_fixed,first_violation_own_displacement_denominator=first_own,scatter_bitwise_equal_flat=True,failed_constraint_indices=failed,stages=stages,error_decomposition=error_report,matrix_vs_direct_dot_max_error=matrix_check,optimizer_scheduler_rng_preserved_after_correction=preserved,native_optimizer_increment=proposal['candidate']['optimizer_step']-base['optimizer_step'],native_scheduler_increment=proposal['candidate']['scheduler_step']-base['scheduler_step'],seconds=time.perf_counter()-begin,formal_training_resumed=False)
  g.restore(algo,base);result['source_restored']=g.compare_states(base,g.capture(algo));assert all(result['source_restored'].values());assert sha(statepath)==sourcehash
  write(out/'RESULT.json',result);return result
 finally:
  g.restore(algo,base)

def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.set_num_interop_threads(1);g.init_ddp(Path('/root/qp_corrected_update_fidelity_v1')/f'forensic_ddp_{os.getpid()}')
 manifest=read(SOURCE/'START_MANIFEST.json')['starts'];freeze=read(SOURCE/'EXECUTION_FREEZE.json');assert all(sha(ROOT/p)==v for p,v in freeze['files'].items())
 protocol=dict(head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),targets=TARGETS,formal_training_resumed=False,formal_solver_modified=False,kkt_tolerance=1e-8,writeback_tolerance=5e-6,definitions={'d64':'u32 promoted to binary64 minus H64 @ lambda64','d32':'formal flat float32 u32 - float32(H64 @ lambda64)','dscatter':'layout-only per-parameter reconstruction of u32 minus reshaped c32, then concatenate; formal code scatters correction, not displacement','dwritten':'exact binary64 difference theta_written(float32) - theta_before(float32)','dwritten_float32_report':'formal Workspace.displacement float32 subtraction after physical write'},checks='same fixed ||u32|| denominator at every layer AND existing own displacement norm gate+1e-15; raw positive signs are not automatically failures',source_restore='existing source rolling state must match all frozen model/AdamW/scheduler/RNG/counter/mode witnesses; zero prior batches')
 write(OUT/'PROTOCOL.json',protocol);results=[]
 try:
  for sid in TARGETS:
   write(OUT/'STATUS.json',dict(stage='RUNNING',start_id=sid,pid=os.getpid(),formal_training_resumed=False))
   result=inspect_start(next(x for x in manifest if x['start_id']==sid));results.append(result);gc.collect();torch.cuda.empty_cache()
  assert all(sha(ROOT/p)==v for p,v in freeze['files'].items())
  write(OUT/'SUMMARY.json',dict(status='COMPLETE',results=results,original_experiment_unchanged=True,formal_training_resumed=False))
  report=['# QP writeback representation forensic','', 'Two immutable source states, zero preceding batches. Formal solver and original experiment remain unchanged.','', '|start|stage|fixed native normalized max|own normalized max|own-gate violations|','|---|---|---:|---:|---:|']
  for r in results:
   for v in r['stages']:report.append(f"|{r['start_id']}|{v['stage']}|{v['fixed_u_normalized_max']:.12g}|{v['own_norm_normalized_max']:.12g}|{v['postwrite_gate_violations']}|")
   report.extend(['',f"{r['start_id']}: first fixed-denominator violation={r['first_violation_fixed_native_denominator']}; first actual-norm gate violation={r['first_violation_own_displacement_denominator']}; solver KKT={r['kkt']}; historical failure reproduced."])
  report+=['','dscatter is a layout-only reconstruction using actual parameter slice boundaries. It is not an alternative writeback implementation. dwritten is the exact difference of stored float32 parameters; the additional report stage isolates subtraction rounding.','Both normalization schemes are shown to avoid attributing a denominator change to rounding. Raw tiny positive inner products are reported separately from tolerance failures.','No threshold modification, repair, extra fallback, phase2, or formal training was executed.']
  (OUT/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n');write(OUT/'STATUS.json',dict(stage='COMPLETE',formal_training_resumed=False))
 except Exception:
  write(OUT/'FAILED.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False));raise
 finally:
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()
