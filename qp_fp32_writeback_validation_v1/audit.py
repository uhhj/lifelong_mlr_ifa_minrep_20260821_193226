"""Independent CPU verification and evidence-derived Stage A conclusion."""
import numpy as np,pandas as pd
from .common import *

def main():
 comp=pd.read_csv(OUT/'MATERIALIZATION_COMPARISON.csv');decomp=pd.read_csv(OUT/'CONSTRAINT_ERROR_DECOMPOSITION.csv');margin=pd.read_csv(OUT/'ROUNDING_MARGIN_ANALYSIS.csv');ulps=read(OUT/'ULP_ANALYSIS.json');evidence=[]
 for sid in TARGETS:
  q=np.load(OUT/sid/'QP_AND_RESIDUALS.npz');sol=read(OUT/sid/'SOLUTION.json');x=np.array(sol['normalized_x']);r=q['b']-q['A']@x;kkt=max(np.maximum(r,0).max(initial=0),np.maximum(-x,0).max(initial=0),np.abs(x*r).max(initial=0));assert kkt<=1e-8
  hn=np.sqrt(np.diag(q['K']));den=np.maximum(hn*float(q['exact_d0_norm']),1e-30);rows=[]
  for method in ['W0','W1','W2']:
   saved=comp[(comp.state_id==sid)&(comp.method==method)].iloc[0];v=decomp[(decomp.state_id==sid)&(decomp.method==method)].sort_values('constraint');actual=q[method];rho=actual/den
   assert np.isclose(np.maximum(rho,0).max(),saved.max_positive_residual,rtol=1e-12,atol=1e-15);assert int((rho>5e-6).sum())==saved.violating_count
   assert np.max(abs(v.r_written-v.r_ideal-v.delta_r))<1e-13;assert np.max(abs(v.r_written-actual))<1e-13
   checks=read(OUT/sid/'STATE_CHECKS.json')[method];assert all(checks['nonmodel_state_unchanged'].values()) and all(checks['full_restore'].values()) and checks['deterministic']
   rows.append(dict(method=method,max_positive_residual=float(rho.max()),physical_pass=bool(rho.max()<=5e-6)))
  for label in ['correction','final_displacement']:
   u=ulps[sid][label];assert sum(u['counts'])==u['total'];assert sum(u['nonzero_counts'])==u['nonzero']
  d=decomp[(decomp.state_id==sid)&(decomp.method=='W1')&decomp.physical_failed];m=margin[(margin.state_id==sid)&(margin.method=='W1')];near=m[m.near_active|m.active];u=ulps[sid]
  evidence.append(dict(state_id=sid,kkt=float(kkt),methods=rows,nonzero_correction_below_half_ulp=u['correction']['nonzero_counts'][0]/u['correction']['nonzero'],nonzero_final_displacement_at_least_4ulp=u['final_displacement']['nonzero_counts'][4]/u['final_displacement']['nonzero'],top20_absolute_share_min=float(d.top20_absolute_share.min()),top20_absolute_share_max=float(d.top20_absolute_share.max()),top20_signed_over_net_min=float(d.top20_signed_over_net.min()),top20_signed_over_net_max=float(d.top20_signed_over_net.max()),active_constraints=int(m.active.sum()),near_active_constraints=int(m.near_active.sum()),near_or_active_margin_below_bound=int(near.margin_below_bound.sum()),all_failed_have_margin_below_bound=bool(m[m.physical_failed].margin_below_bound.all()),largest_margin_to_bound_ratio_near=float(near.margin_over_bound.max())))
 freeze=read(OUT/'FREEZE.json');unchanged={p:sha(ROOT/p)==v for p,v in freeze['files'].items()};assert all(unchanged.values())
 write(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',states=evidence,frozen_inputs_unchanged=unchanged,formal_training_resumed=False))
 test=read(OUT/'UNIT_TEST_REPORT.json');test['real_state_tests']={s:read(OUT/s/'STATE_CHECKS.json') for s in TARGETS};write(OUT/'UNIT_TEST_REPORT.json',test)
 lines=['','## Measured conclusions','']
 for v in evidence:
  lines += [f"- {v['state_id']}: continuous KKT={v['kkt']:.12g}; {100*v['nonzero_correction_below_half_ulp']:.4f}% of nonzero correction coordinates are below0.5ULP of the source parameter. {100*v['nonzero_final_displacement_at_least_4ulp']:.4f}% of nonzero final displacement coordinates are >=4 source-parameter ULP.",f"  Failed constraints' top20 absolute-contribution shares are {100*v['top20_absolute_share_min']:.4f}%–{100*v['top20_absolute_share_max']:.4f}%. Their signed top20/net-error ratios range from {v['top20_signed_over_net_min']:.5g} to {v['top20_signed_over_net_max']:.5g}. This does not support a few-coordinate-dominance explanation.",f"  {v['near_or_active_margin_below_bound']}/{v['near_active_constraints']} near-active constraints have ideal margin below the requested rounding bound; all actual failures are in that set. Largest near-active margin/bound={v['largest_margin_to_bound_ratio_near']:.8g}."]
 lines += ['', 'Removing native-then-correction materialization does not repair either state: W1 and W2 both retain the original1/9 failed constraints. The fixed target has insufficient slack for its nearest-FP32 representation. DOUBLE_MATERIALIZATION is not supported as the principal explanation on these states.', '', 'Target rounding is determined by target ULP, which can differ from source ULP after a binade change. Error measured in source ULP can therefore exceed0.5 without indicating a faulty FP32 rounding operation. Bounds use target spacing, and the maximum of upward/downward spacing is also retained.', '', '|state|W1 seconds including readback|W1 full double target MiB|W1 peak GPU increment MiB|W0 peak GPU increment MiB|', '|---|---:|---:|---:|---:|']
 for sid in TARGETS:
  w=comp[(comp.state_id==sid)&(comp.method=='W1')].iloc[0];a=comp[(comp.state_id==sid)&(comp.method=='W0')].iloc[0];lines.append(f"|{sid}|{w.materialization_seconds:.6g}|{w.full_float64_target_bytes/2**20:.3f}|{w.incremental_gpu_peak_bytes/2**20:.3f}|{a.incremental_gpu_peak_bytes/2**20:.3f}|")
 lines += ['', 'These two timings include diagnostic rereading of actual displacement and are not a throughput benchmark. Memory is affordable on the tested3090 for StageA, but W1 fails physical feasibility; no120-state aggregate cost was measured.', '', 'Final answers8–13: NOT_RUN/NOT_REEVALUATED by the frozen StageA stopping rule, including all120-state gates, old/new NLL reevaluation, trajectory-disjoint evaluation and changes to the118 earlier successful points. Historical results remain immutable.', '', 'Next step is a separately preregistered physical-feasibility study, not GN workset. No candidate P1/P2/P3 was run. Continuous solver/QP semantics do not need modification on the evidence here; a validated physical realization is still required. This is not a claim that the FP32 feasible set is empty (zero displacement remains feasible).']
 path=OUT/'FINAL_REPORT.md';path.write_text(path.read_text()+'\n'.join(lines)+'\n')
 summary=read(OUT/'SUMMARY.json');summary['status']='COMPLETE_STOP_AFTER_STAGE_A';summary['independent_audit']='PASS';write(OUT/'SUMMARY.json',summary)
 from PIL import Image
 for p in OUT.glob('*.png'):
  with Image.open(p) as im:im.verify()
if __name__=='__main__':main()
