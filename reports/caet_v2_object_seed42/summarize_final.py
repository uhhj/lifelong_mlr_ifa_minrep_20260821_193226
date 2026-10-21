"""Read-only final task-level comparison and cost disclosure."""
import json,sys,datetime
from fractions import Fraction as F
from pathlib import Path
p=Path(sys.argv[1]);a=json.loads((p/'LOCAL_FINAL_AUDIT.json').read_text());assert a['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS'
mat={(r['stage'],r['task']):r for r in a['matrix']};out={}
for method in ('ra','v1','v2'):
 out[method]={}
 for t in range(6,10):
  diag=F(mat[t,t][method]);final=F(mat[9,t][method]);peak=max(F(mat[s,t][method]) for s in range(t,10))
  out[method][str(t)]=dict(diagonal_percent=float(diag*100),final_percent=float(final*100),diagonal_minus_final_pp=float((diag-final)*100),peak_minus_final_pp=float((peak-final)*100))
cost={}
for t in (8,9):
 root=p/'current/effects/models'/('task%d'%t);fit=json.loads((root/'ACCEPTANCE.json').read_text());h=json.loads((root/'epochs.json').read_text())
 assert len(h)==10 and sum(r['updates'] for r in h)==fit['effect_updates']
 cost[str(t)]=dict(optimizer_updates=fit['effect_updates'],fit_seconds=fit['seconds'],anchor_exposures=sum(r['anchor_windows'] for r in h),total_window_exposures=sum(r['total_window_exposures'] for r in h),pair_exposures=sum(r['graph_pair_exposures'] for r in h))
constraints={}
for t in (8,9):
 previous=None;peak=0.;violations=0
 for ep in range(100):
  c=json.loads((p/'current/ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,ep))).read_text())
  assert all(v==0 for v in c['mu_before'].values()) if previous is None else previous==c['mu_before']
  previous=c['mu_after'];peak=max(peak,max(previous.values(),default=0))
  violations+=sum(c['values'][k]>c['bounds'][k]+.05 for k in c['bounds'])
 last=c;constraints[str(t)]=dict(max_mu=peak,violated_group_epoch_count=violations,final_violated_groups=sum(last['values'][k]>last['bounds'][k]+.05 for k in last['bounds']),groups=len(last['bounds']))
result=dict(status='TASK_AND_COST_RECOMPUTATION_PASS',tasks=out,effect_cost=cost,constraints=constraints,main_epoch_seconds=a['main_training_seconds'],evaluation_vector_seconds=a['formal']['mean_vector_seconds']*20,historical_reference_scoring_seconds='Not independently timed; do not invent an isolated cost')
(p/'TASK_COST_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
report=(p/'REPORT.md').read_text(encoding='utf8')
report+='\n## Task-level interpretation\n\nFWT uses the frozen mean trained-diagonal success definition, not untrained-task zero-shot forward transfer. V2 improves FWT by 2.5pp versus RA and 3.75pp versus v1; it is not an across-task uniform gain. The RA-to-v2 Task8 diagonal falls 90% to80%, while Task9 rises60% to80%; Task6/7 stay30%/100%.\n\n|Task|RA diagonal|v1 diagonal|v2 diagonal|RA final|v1 final|v2 final|\n|---|---:|---:|---:|---:|---:|---:|\n'
for t in range(6,10):
 vals=[out[m][str(t)]['diagonal_percent'] for m in ('ra','v1','v2')]+[out[m][str(t)]['final_percent'] for m in ('ra','v1','v2')]
 report+='|'+str(t)+'|'+'|'.join(f'{v:.1f}' for v in vals)+'|\n'
report+='\nRelative to RA, final old-task changes are Task6 +10pp, Task7 -10pp, Task8 +5pp: net +1 successful episode across60 old-task episodes (+1.6667pp). Relative to v1 they are -35pp/-15pp/0pp: old-task mean -16.6667pp. Thus the surrogate protection did not establish uniformly improved retention. Task6 still falls from its best measured80% to20% (60pp); Task7 falls from100% to80% (20pp). NBT is lower/better than RA by2.7778pp, but worse than v1 by11.1111pp; negative NBT can coexist with severe later loss because this metric averages changes relative to each task first-learned checkpoint, including earlier improvements.\n\nNo attribution to transfer alone or protection alone is established: causal rebinding, supported transfer, conflict gating and functional retention changed together, and Task7 intentionally falls back. V1 post-action-input leakage prevents treating its higher retention as validated causal-effect transfer, but its observed scores remain intact. Single seed,20 episodes per cell and global action RNG preclude strong generalization claims.\n'
report+='\n## Cost and constraint disclosure\n\nMain105500 optimizer updates,8 isolated preflight updates,1820 additional predictor updates. Main summed epoch time %.2f minutes; completed evaluation vector time %.2f minutes. Epoch time includes auxiliary policy gradients and end-of-epoch full-group scoring; historical-reference setup is not separately timed.\n'%(a['main_training_seconds']/60,a['formal']['mean_vector_seconds']*20/60)
report+='\n```json\n'+json.dumps(dict(effect_fit_cost=cost,final_constraints=constraints),indent=2)+'\n```\n'
(p/'REPORT.md').write_text(report,encoding='utf8');print(json.dumps(result,indent=2))
