"""Analyze saved S/C scalars and cache metadata only; no model or remote calls."""
from pathlib import Path
import json,hashlib,statistics
R=Path(__file__).resolve().parent;W=R.parent;D=R/'final_readonly_20261027_222119';A=R/'s_learning_analysis_20261027';A.mkdir(exist_ok=True);sources={}
def read(p):
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf8'))
c=read(D/'COMPARISON.json');assert c['status']=='PASS_ACCEPTED_COMPARISON'
results={};all_ratios=[]
for suite in ['GOAL','OBJECT']:
 for seed in [41,42,43]:
  case=f'{suite}_{seed}'
  root=W/'20261025_ccbp_v4_full/final_readonly_20261026_014654' if case=='GOAL_41' else W/('20261026_ccbp_v4_s_goal42_43/final_readonly_20261026_164119' if suite=='GOAL' else '20261026_ccbp_v4_s_object41_42_43/final_readonly_20261027_120754')
  prefix='' if case=='GOAL_41' else f'seed{seed}/';light=root/'light' if not prefix else root/f'seed{seed}/light';ev=read(root/'EVIDENCE.json');assert read(root/'LOCAL_FINAL_AUDIT.json')['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
  def load(n):
   p=light/n;x=read(p);meta=ev['light'][prefix+n]
   assert hashlib.sha256(meta['text'].encode()).hexdigest()==meta['sha256'] and x==json.loads(meta['text']);return x
  g=load('GRADIENTS_S.json');ep=load('EPOCHS_S.json');assert len(g)==12
  mem={}
  for task in [6,7,8]:
   m=load(f'MEMORY_S_{task}.json');mem[str(task)]=dict(count=m['count'],collection_successes=m['collection_successes'],unique_trajectories=len({x['trajectory'] for x in m['rows']}),length_distribution={str(n):sum(x['length']==n for x in m['rows']) for n in range(1,9)},scales=m['scales'])
  stages={}
  for st in [7,8,9]:
   gg=[x for x in g if x['stage']==st];assert {x['epoch'] for x in gg}=={0,9,49,99}
   ratios=[(x['retain_weighted']['l2']+x['retain_route_weighted']['l2'])/x['bc']['l2'] for x in gg];all_ratios+=ratios
   ee=[x for x in ep if x['stage']==st];n=sum(x['batches'] for x in ee)
   ex=sum(.1*x['totals']['expert_loss']+.01*x['totals']['expert_route'] for x in ee)/n
   retain=sum(.1*x['totals']['retain_loss']+.01*x['totals']['retain_route'] for x in ee)/n
   slots=sum(x['totals']['anchor_slots'] for x in ee);actual=sum(x['totals']['anchor_actual'] for x in ee);assert slots==4*n and actual==slots
   stages[str(st)]=dict(sparse_gradient_points=[dict(epoch=x['epoch'],retention_sum_norm_upper_over_mixed_BC=r,retain_norm=x['retain_weighted']['l2'],route_norm=x['retain_route_weighted']['l2'],mixed_BC_norm=x['bc']['l2']) for x,r in zip(gg,ratios)],retention_route_active_points=sum(x['retain_route_weighted']['l2']>0 for x in gg),full_epoch_mean_weighted_expert_aux=ex,full_epoch_mean_weighted_retain_aux=retain,retention_to_expert_loss_value_ratio=retain/ex,anchor_slots=slots,actual_slots=actual,expected_task6_draws_per_cached_window=n*4/(st-6)/mem['6']['count'])
  cc=c['C'][case];ss=c['S'][suite][str(seed)]
  acquisition={str(t):dict(C=cc['matrix'][f'{t}_{t}']['percent'],S=ss['matrix'][f'{t}_{t}']['percent'],S_minus_C=ss['matrix'][f'{t}_{t}']['percent']-cc['matrix'][f'{t}_{t}']['percent']) for t in range(6,10)}
  results[case]=dict(acquisition=acquisition,memory=mem,stages=stages)
assert len(all_ratios)==72
out=dict(status='PASS_SAVED_DATA_ONLY_RECOMPUTATION',cases=results,gradient_summary=dict(points=72,median_sum_norm_upper_over_mixed_BC=statistics.median(all_ratios),max_sum_norm_upper_over_mixed_BC=max(all_ratios),no_gradient_angles_saved=True,no_current_task_only_gradient_saved=True,no_layerwise_gradient_saved=True),source_sha256=sources,limits=['Sparse first eligible batch at epochs0/9/49/99, not all optimization steps','Norm sums bound retention combined norm but do not identify its direction','BC mixes old/new samples; no new-task-only loss or gradient separation','Offline expert diagnostics cover old tasks, not current Task9 deployment states','Different full training trajectories: not a common Stage8 checkpoint fork'],new_model_calls=0,new_optimizer_calls=0,new_environments=0,remote_calls=0)
(A/'AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(dict(status=out['status'],gradient_summary=out['gradient_summary'],stage9={k:dict(loss_ratio=v['stages']['9']['retention_to_expert_loss_value_ratio'],route_active=v['stages']['9']['retention_route_active_points'],task6_cache=v['memory']['6']['count'],task9=v['acquisition']['9']) for k,v in results.items()}),indent=2))
