"""Descriptive report answers from frozen completed measurements, no selection."""
import numpy as np
from .common import *

def stats(v):
 v=[float(x) for x in v if x is not None and np.isfinite(x)]
 return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),min=float(np.min(v)),max=float(np.max(v))) if v else dict(n=0)

def describe(starts,eng,geo,effects,inter,ev,strata,allmeans,choices,behavior):
 energy=dict(rho_head=stats([x['rho_head'] for x in eng]),rho_all=stats([x['rho_all'] for x in eng]),conditional_increment_norm=stats([x['conditional_increment_norm'] for x in eng]),head_fraction_of_increment_energy=stats([x['conditional_head_norm']**2/x['conditional_increment_norm']**2 if x['conditional_increment_norm'] else None for x in eng]),update_GN_energy_fraction=stats([x['E_r']/x['E_full'] if x['E_full'] else None for x in eng]),tail_GN_energy_fraction=stats([x['E_tail']/x['E_full'] if x['E_full'] else None for x in eng]),cross_GN_energy_fraction=stats([x['E_cross']/x['E_full'] if x['E_full'] else None for x in eng]),trace_top16=stats([x['coverage16'] for x in geo]))
 absolute={};interaction={}
 for role in ['T_old','T_new']:
  rr=[r for r in effects if r['role']==role]
  absolute[role]={k:stats([r[k] for r in rr]) for k in ['marginal_with_rest','absolute_keep','absolute_remove']}
  absolute[role]['keep_absolute_worse_starts']=sum(r['absolute_keep']>r['tau'] for r in rr)
  absolute[role]['remove_absolute_worse_starts']=sum(r['absolute_remove']>r['tau'] for r in rr)
  ii=[r for r in inter if r['bank']=='T' and r['role']==role]
  interaction[role]=dict(value=stats([r['interaction'] for r in ii]),absolute_size=stats([abs(r['interaction']) for r in ii]),raw_sign_reversals_with_vs_without=sum(r['marginal_with_rest']*r['marginal_without_rest']<0 for r in ii),warning='Raw sign reversals descriptive; see numerical bands and trajectory intervals for discernibility')
 costs={m:dict(joint_gain=stats([r['gain'] for r in ev if r['method']==m]),old_cost=stats([r['old_cost'] for r in ev if r['method']==m]),new_cost=stats([r['new_cost'] for r in ev if r['method']==m])) for m in METHODS}
 monitor=[];gripper=[]
 for s in starts:
  sid=s['start_id'];a=allmeans[sid]['T'];chosen=choices[sid]['GN']['candidate']
  for role in sorted({k[0] for k in a if k[0].startswith('T_monitor')}):
   ds=sorted({k[2] for k in a if k[0]==role})
   monitor.append(dict(start_id=sid,role=role,candidate=chosen,vs_KEEP=float(np.mean([a[role,chosen,d]-a[role,'KEEP',d] for d in ds])),vs_BASE=float(np.mean([a[role,chosen,d]-a[role,'BASE',d] for d in ds]))))
  gripper.extend([r for r in strata if r['start_id']==sid and r['stratum']=='gripper_change' and r['candidate']==chosen])
 paired_behavior=[]
 for b in behavior.get('cell_values',[]):
  s=next(s for s in starts if s['start_id']==b['point']);role='T_old' if b['task']==s['task_id']-1 else 'T_new'
  r=next(r for r in effects if r['start_id']==b['point'] and r['role']==role)
  paired_behavior.append(dict(start_id=b['point'],task=b['task'],remove_minus_keep_success=b['difference'],remove_minus_keep_NLL=-r['marginal_with_rest']))
 result=dict(engagement_and_energy=energy,absolute_and_marginal_effects=absolute,interactions=interaction,D_selected_method_costs=costs,other_old_task_rows=monitor,worst_other_old_task_vs_KEEP=max((r['vs_KEEP'] for r in monitor),default=None),selected_gripper_change_rows=gripper,behavior_NLL_paired=paired_behavior,formal_training_resumed=False)
 write(OUT/'DESCRIPTIVE_ANSWERS.json',result)
 return result
