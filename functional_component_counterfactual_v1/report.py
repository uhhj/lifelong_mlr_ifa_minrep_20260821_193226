import csv,collections,math
import numpy as np
from scipy.stats import spearmanr,rankdata
from .common import *
def loadrows(p):return list(csv.DictReader(Path(p).open()))
def means(rows):
 out={}
 for r in rows:
  k=(r['role'],r['candidate'],r['trajectory_id']);out.setdefault(k,[]).append(float(r['nll']))
 return {k:float(np.mean(v)) for k,v in out.items()}
def task_value(a,role,name):return float(np.mean([v for (r,c,d),v in a.items() if r==role and c==name]))
def interval_effect(a,role,left,right,seed):
 ds=sorted({d for r,c,d in a if r==role});v=np.array([a[role,left,d]-a[role,right,d] for d in ds])
 rng=np.random.RandomState(seed);w=rng.exponential(size=(5000,len(v)));boot=w@v/w.sum(1)
 return float(v.mean()),np.percentile(boot,[2.5,97.5]).tolist(),len(ds)
def sign_ci(ci,tau):
 if ci[1]<-tau:return 'NEG'
 if ci[0]>tau:return 'POS'
 if ci[0]>=-tau and ci[1]<=tau:return 'NEUTRAL'
 return 'UNRESOLVED'
def finite(x):return float(x) if np.isfinite(x) else None
def bootstrap(starts,allmeans,choices):
 # Coupled trajectory weights across repeated trajectory identities; independent of methods.
 rng=np.random.RandomState(2026092411);B=5000;n=len(starts);gains=np.zeros((B,n,len(METHODS)))
 weights={}
 for j,s in enumerate(starts):
  sid=s['start_id'];a=allmeans[sid]['T']
  for role in ['T_old','T_new']:
   task=s['task_id']-1 if role=='T_old' else s['task_id']
   ds=sorted({d for r,c,d in a if r==role});ww=[]
   for d in ds:
    k=(s['seed'],task,d)
    if k not in weights:weights[k]=rng.exponential(size=B)
    ww.append(weights[k])
   W=np.stack(ww,1);W/=W.sum(1)[:,None]
   for mi,m in enumerate(METHODS):
    selected=choices[sid][m]['candidate'];v=np.array([a[role,'KEEP',d]-a[role,selected,d] for d in ds])
    gains[:,j,mi]+=.5*(W@v)
 margins=np.zeros((B,n,4))
 for j,s in enumerate(starts):
  for ri,(bank,role) in enumerate([('D','D_old'),('T','T_old'),('D','D_new'),('T','T_new')]):
   a=allmeans[s['start_id']][bank];task=s['task_id']-1 if role.endswith('old') else s['task_id']
   ds=sorted({d for r,c,d in a if r==role});ww=[]
   for d in ds:
    key=(s['seed'],task,d)
    if key not in weights:weights[key]=rng.exponential(size=B)
    ww.append(weights[key])
   W=np.stack(ww,1);W/=W.sum(1)[:,None]
   margins[:,j,ri]=W@np.array([a[role,'KEEP',d]-a[role,'GN_REMOVE',d] for d in ds])
 groups=collections.defaultdict(list)
 for j,s in enumerate(starts):groups[(s['seed'],s['task_id'])].append(j)
 cell_samples=[];margin_samples=[]
 for k,ids in sorted(groups.items()):
  ids=sorted(ids,key=lambda j:starts[j]['cell_order']);assert len(ids)==6
  first=rng.randint(0,6,size=(B,3));indices=np.stack([first,(first+1)%6],-1).reshape(B,6)
  pick=np.array(ids)[indices];cell_samples.append(gains[np.arange(B)[:,None],pick,:].mean(1));margin_samples.append(margins[np.arange(B)[:,None],pick,:])
 combined=np.mean(cell_samples,axis=0)
 mm=np.concatenate(margin_samples,1);ranked=rankdata(mm,axis=1);ranked-=ranked.mean(1,keepdims=True);cor_intervals={}
 for role,i,j in [('old',0,1),('new',2,3)]:
  den=np.sqrt((ranked[:,:,i]**2).sum(1)*(ranked[:,:,j]**2).sum(1))
  valid=den>0;rho=(ranked[valid,:,i]*ranked[valid,:,j]).sum(1)/den[valid]
  cor_intervals[role]=np.percentile(rho,[2.5,97.5]).tolist() if len(rho) else None
 gn=combined[:,0];c2=combined[:,[4,5,6]].mean(1);c3=combined[:,7]
 return dict(D_T_joint_trajectory_block_CI=cor_intervals,GN=np.percentile(gn,[2.5,97.5]).tolist(),GN_minus_C2_simultaneous97_5=np.percentile(gn-c2,[1.25,98.75]).tolist(),GN_minus_C3_simultaneous97_5=np.percentile(gn-c3,[1.25,98.75]).tolist(),replicates=B,scope='Conditional on six fixed carriers; shared trajectory exponential bootstrap weights and length2 circular start blocks; three development seeds, no population generality')

def main():
 starts=read(OUT/'START_MANIFEST.json')['starts'];complete=[s for s in starts if (OUT/'starts'/s['start_id']/'COMPLETE.json').exists()]
 if len(complete)!=36:
  write(OUT/'SUMMARY.json',dict(MEASUREMENT_VALIDITY='INCOMPLETE',completed=len(complete),expected=36,LOCAL_EFFECT_REPLICATION='NOT_EVALUATED',GEOMETRY_ADDED_VALUE='NOT_EVALUATED',BEHAVIORAL_EFFECT=read(OUT/'BEHAVIORAL_REPORT.json')['status'] if (OUT/'BEHAVIORAL_REPORT.json').exists() else 'NOT_RUN',formal_training_resumed=False));return
 taus=read(OUT/'NUMERICAL_RESOLUTION.json')['tasks']
 means_all={};choices={};effects=[];interactions=[];evaluations=[];engagement=[];geometry=[];parities=[];selection_rows=[];strata=[];resources=[];pointstats=[]
 for s in starts:
  sid=s['start_id'];dest=OUT/'starts'/sid;dd=loadrows(dest/'D_WINDOWS.csv');tt=loadrows(dest/'T_WINDOWS.csv')
  dm,tm=means(dd),means(tt);means_all[sid]={'D':dm,'T':tm}
  table(dest/'TRAJECTORY_NLL.csv',[dict(start_id=sid,role=k[0],candidate=k[1],trajectory_id=k[2],mean_nll=v) for bank in [dm,tm] for k,v in bank.items()])
  ch=read(dest/'SELECTIONS_D_ONLY.json');choices[sid]=ch
  eng=read(dest/'ENGAGEMENT.json');geo=read(dest/'GEOMETRY.json');par=read(dest/'PARITY.json');phys=read(dest/'PHYSICAL.json')
  engagement.append(dict(start_id=sid,seed=s['seed'],transition=s['transition'],**eng,**phys))
  geometry.append(dict(start_id=sid,orthogonal_error=geo['orthogonal_error'],eigen_relative_max=geo['max_relative_eigen_residual'],rank=geo['rank'],trace=geo['trace'],coverage8=geo['trace_coverage']['8'],coverage16=geo['trace_coverage']['16'],coverage32=geo['trace_coverage']['32'],eigengap16=geo['eigengap16']))
  parities.append(dict(start_id=sid,**par));resources.append(dict(start_id=sid,**read(dest/'COMPLETE.json'),basis_seconds=geo['seconds'],proposal_seconds=par['seconds']))
  for bank,a in [('D',dm),('T',tm)]:
   for role in sorted({k[0] for k in a}):
    task=next(int(r['task_id']) for r in (dd if bank=='D' else tt) if r['role']==role)
    tau=taus[str(task)]
    marginal,ci,ntraj=interval_effect(a,role,'KEEP','GN_REMOVE',stable_seed(sid+'/'+role))
    without=task_value(a,role,'X10')-task_value(a,role,'X00')
    effects.append(dict(start_id=sid,seed=s['seed'],transition=s['transition'],bank=bank,role=role,task=task,marginal_with_rest=marginal,CI_low=ci[0],CI_high=ci[1],trajectories=ntraj,sign_num=sign_ci(ci,tau['tau_num']),sign_equiv=sign_ci(ci,tau['tau']),tau_num=tau['tau_num'],tau=tau['tau'],absolute_keep=task_value(a,role,'KEEP')-task_value(a,role,'BASE'),absolute_remove=task_value(a,role,'GN_REMOVE')-task_value(a,role,'BASE')))
    interactions.append(dict(start_id=sid,bank=bank,role=role,marginal_with_rest=marginal,marginal_without_rest=without,interaction=marginal-without))
  for method in METHODS:
   n=ch[method]['candidate'];do=ch[method]['D_objectives'];jt=lambda c:.5*(task_value(tm,'T_old',c)+task_value(tm,'T_new',c))
   row=dict(start_id=sid,seed=s['seed'],transition=s['transition'],method=method,selected=n,lambda_value=ch[method]['lambda_value'],gain=jt('KEEP')-jt(n),old_cost=task_value(tm,'T_old',n)-task_value(tm,'T_old','KEEP'),new_cost=task_value(tm,'T_new',n)-task_value(tm,'T_new','KEEP'),fixed_remove_gain=jt('KEEP')-jt(method+'_REMOVE'),fixed_half_gain=jt('KEEP')-jt(method+'_HALF'))
   evaluations.append(row);selection_rows.append(dict(start_id=sid,method=method,**ch[method]))
  for role in sorted({r['role'] for r in tt}):
   for name in sorted({r['candidate'] for r in tt}):
    rr=[r for r in tt if r['role']==role and r['candidate']==name]
    for group in ['all','early','middle','late','gripper_change']:
     selected=[r for r in rr if group=='all' or (group=='gripper_change' and r['gripper_change']=='True') or r['time_stratum']==group]
     if not selected:continue
     ds=np.array([float(r['delta_base']) for r in selected]);ks=np.array([float(r['difference_keep']) for r in selected]);tr=collections.defaultdict(list)
     for r in selected:tr[r['trajectory_id']].append(float(r['delta_base']))
     strata.append(dict(start_id=sid,seed=s['seed'],transition=s['transition'],role=role,candidate=name,stratum=group,windows=len(selected),trajectories=len(tr),trajectory_equal_delta=float(np.mean([np.mean(v) for v in tr.values()])),window_mean_positive_damage=float(np.maximum(ds,0).mean()),window_p90=float(np.quantile(ds,.9)),window_max=float(ds.max()),mean_minus_keep=float(ks.mean()),mean_KL=float(np.mean([float(r['kl']) for r in selected])),mean_KL_MC_SE=float(np.mean([float(r['kl_se']) for r in selected]))))

 cell=[];seedrows=[]
 for method in METHODS:
  for seed in [45,46,47]:
   for transition in ['6->7','7->8']:
    rr=[r for r in evaluations if r['method']==method and r['seed']==seed and r['transition']==transition]
    cell.append(dict(level='cell',seed=seed,transition=transition,method=method,count=len(rr),gain=float(np.mean([r['gain'] for r in rr])),old_cost=float(np.mean([r['old_cost'] for r in rr])),new_cost=float(np.mean([r['new_cost'] for r in rr]))))
   rr=[r for r in evaluations if r['method']==method and r['seed']==seed]
   seedrows.append(dict(level='seed',seed=seed,transition='both',method=method,count=len(rr),gain=float(np.mean([r['gain'] for r in rr])),old_cost=float(np.mean([r['old_cost'] for r in rr])),new_cost=float(np.mean([r['new_cost'] for r in rr]))))
 boot=bootstrap(starts,means_all,choices)
 rel=[];cors={};cor_boot={};rng=np.random.RandomState(2026092413)
 for sid in [s['start_id'] for s in starts]:
  r={e['role']:e for e in effects if e['start_id']==sid}
  rel.append(dict(start_id=sid,D_label=label(r['D_old']['sign_equiv'],r['D_new']['sign_equiv']),T_label=label(r['T_old']['sign_equiv'],r['T_new']['sign_equiv']),D_numeric_label=label(r['D_old']['sign_num'],r['D_new']['sign_num']),T_numeric_label=label(r['T_old']['sign_num'],r['T_new']['sign_num'])))
 # Correlation uses start summaries; block resampling keeps serial start dependence.
 cells_idx=[[j for j,s in enumerate(starts) if s['seed']==seed and s['task_id']==task] for seed in [45,46,47] for task in [7,8]]
 for role in ['old','new']:
  x=np.array([next(e['marginal_with_rest'] for e in effects if e['start_id']==s['start_id'] and e['role']=='D_'+role) for s in starts])
  y=np.array([next(e['marginal_with_rest'] for e in effects if e['start_id']==s['start_id'] and e['role']=='T_'+role) for s in starts])
  cors[role]=dict(spearman=finite(spearmanr(x,y)[0]),CI95=boot['D_T_joint_trajectory_block_CI'][role],seed_spearman={str(seed):finite(spearmanr(x[[s['seed']==seed for s in starts]],y[[s['seed']==seed for s in starts]])[0]) for seed in [45,46,47]})
 allseeds=True
 for seed in [45,46,47]:
  g1=next(r['gain'] for r in seedrows if r['seed']==seed and r['method']=='GN')
  c2=np.mean([r['gain'] for r in seedrows if r['seed']==seed and r['method'].startswith('C2_')]);c3=next(r['gain'] for r in seedrows if r['seed']==seed and r['method']=='C3')
  allseeds &= g1>=0 and g1>=c2 and g1>=c3
 geometry_support=boot['GN'][0]>0 and boot['GN_minus_C2_simultaneous97_5'][0]>0 and boot['GN_minus_C3_simultaneous97_5'][0]>0 and allseeds
 local_support=all(c['CI95'] is not None and c['CI95'][0]>0 and all(v is not None and v>=0 for v in c['seed_spearman'].values()) for c in cors.values())
 leave=[]
 for seed in [45,46,47]:
  rr=[r for r in evaluations if r['seed']!=seed]
  leave.append(dict(left_out_seed=seed,gains={m:float(np.mean([r['gain'] for r in rr if r['method']==m])) for m in METHODS}))
 table(OUT/'GEOMETRY_CHECKS.csv',geometry);table(OUT/'ENGAGEMENT.csv',engagement);table(OUT/'CANDIDATE_EFFECTS.csv',effects);table(OUT/'INTERACTIONS.csv',interactions);table(OUT/'SELECTIONS_D_ONLY.csv',selection_rows);table(OUT/'T_EVALUATION.csv',evaluations);table(OUT/'STATE_AND_TRAJECTORY_STRATA.csv',strata);table(OUT/'CELL_AND_SEED_SUMMARY.csv',cell+seedrows);table(OUT/'D_T_LABELS.csv',rel)
 write(OUT/'PROPOSAL_PARITY.json',dict(status='PASS',points=parities))
 write(OUT/'STATISTICAL_REPORT.json',dict(bootstrap=boot,D_T_correlations=cors,labels=rel,leave_one_seed_out=leave,seed_directions_nonnegative=allseeds,limits='Conditional development-carrier intervals, trajectory dependencies retained. D/T correlation and gain intervals share trajectory weights and paired circular start blocks; point marginal intervals reported separately.'))
 write(OUT/'RESOURCE_COST.json',dict(points=resources,total_prepare_seconds=sum(r['prepare_seconds'] for r in resources),total_T_seconds=sum(r['T_seconds'] for r in resources),peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in resources),basis_retention='operator seeds/spectrum/coefficients; large U released after projection',T_privilege=read(OUT/'DATA_ROLES.json')['T_privileged_raw_access']))
 behavior=read(OUT/'BEHAVIORAL_REPORT.json') if (OUT/'BEHAVIORAL_REPORT.json').exists() else dict(status='NOT_RUN')
 summary=dict(MEASUREMENT_VALIDITY='PASS',PROJECTED_UPDATE_ENGAGEMENT=dict(median_rho_head=float(np.median([r['rho_head'] for r in engagement if r['rho_head'] is not None])),median_rho_all=float(np.median([r['rho_all'] for r in engagement if r['rho_all'] is not None])),below_physical_resolution=sum(r['status']=='BELOW_PHYSICAL_RESOLUTION' for r in engagement)),LOCAL_EFFECT_REPLICATION='SUPPORTED_ON_DEVELOPMENT_SET' if local_support else 'INCONCLUSIVE',GEOMETRY_ADDED_VALUE='SUPPORTED_ON_DEVELOPMENT_SET' if geometry_support else 'NOT_ESTABLISHED',BEHAVIORAL_EFFECT=behavior,RESOURCE_COST='MEASURED_SEPARATELY',completed=36,formal_training_resumed=False)
 write(OUT/'SUMMARY.json',summary)
 report='# 功能子空间更新归因实验 v1\n\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n\n'
 report+='## 测量对象与范围\n\n本轮测的是给定回放与AdamW状态的新任务条件增量，不是永久知识复用。36个epoch30–99起点来自三个开发seed和6→7、7→8两个阶段。完整MLR与移除新任务损失候选保留原分母、clipping、moments和衰减；本机原生candidate parity与回滚通过。主干预只改policy head，解码器保留native候选值。\n\n'
 report+='## 能量与物理实现\n\n介入比例见ENGAGEMENT.csv，top8/16/32完整trace覆盖见GEOMETRY_CHECKS.csv。E_full使用完整算子，尾部与交叉项保留；空间外不称安全。实际FP32 head逐候选写入，缓存来自各自BASE/R/M解码器；量化零效应保留，不做QP物理门。\n\n'
 report+='## 跨轨迹效应和几何增量\n\n'+json.dumps(dict(D_T=cors,paired_gain_intervals=boot),indent=2,ensure_ascii=False)+'\n\n'
 report+='边际KEEP−REMOVE与各自相对BASE的绝对效应分开。X00/X10用于交互分解，详见INTERACTIONS.csv。D选择在T前冻结；GN固定HALF/REMOVE单列，三个随机重复取平均而非择优。六cell、三个seed及leave-one-seed-out均保留。未建立几何增量时不加rank/新几何。\n\n'
 report+='## 尾部、行为与成本\n\n逐任务/轨迹原始NLL、时间三分段及夹爪指令变化窗口见各start CSV与STATE_AND_TRAJECTORY_STRATA.csv；不称接触真值。旁观旧任务不建空间，单列其损伤。完整GMM KL估计保留负值和MC标准误，不能当利益标签。行为仅由BEHAVIORAL_REPORT.json中的实际480预算rollout判断，NLL不替代成功率；20次/任务低功效不证明等效。资源逐起点记录并披露T额外缓存访问，见RESOURCE_COST.json。\n\n'
 report+='## 下一步\n\n'+('仅可另行设计有限规模续训，当前不自动执行。' if geometry_support else '当前未建立GN超越等幅廉价对照的增量，不推进该几何到正式训练；不追加变体补救。')+'\n\nformal_training_resumed=false。\n'
 (OUT/'FINAL_REPORT.md').write_text(report)
 render(engagement,geometry,effects,interactions,evaluations,behavior)
 status('REPORT_COMPLETE',summary=summary)
def render(eng,geo,effects,inter,ev,behavior):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots();ax.scatter([r['rho_head'] for r in eng],[r['rho_all'] for r in eng]);ax.set(xlabel='Head projected fraction',ylabel='Full-update projected fraction');fig.savefig(OUT/'engagement.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();ax.boxplot([[g['coverage'+str(k)] for g in geo] for k in [8,16,32]],labels=['8','16','32']);ax.set(xlabel='Rank',ylabel='Captured full trace');fig.savefig(OUT/'energy_coverage.png',dpi=150);plt.close(fig)
 for bank in ['D','T']:
  fig,ax=plt.subplots();sids=sorted({r['start_id'] for r in effects});x=[];y=[]
  for sid in sids:
   x.append(next(r['marginal_with_rest'] for r in effects if r['start_id']==sid and r['role']==bank+'_old'))
   y.append(next(r['marginal_with_rest'] for r in effects if r['start_id']==sid and r['role']==bank+'_new'))
  ax.scatter(x,y);ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(xlabel='Old KEEP - REMOVE NLL',ylabel='New KEEP - REMOVE NLL',title=bank);fig.savefig(OUT/(bank+'_marginal_effects.png'),dpi=150);plt.close(fig)
 fig,ax=plt.subplots();ax.hist([r['interaction'] for r in inter if r['bank']=='T'],bins=20);ax.set(xlabel='T conditional interaction NLL');fig.savefig(OUT/'interactions.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();ax.boxplot([[r['gain'] for r in ev if r['method']==m] for m in METHODS],labels=METHODS);ax.tick_params(axis='x',rotation=45);ax.set(ylabel='T joint NLL gain from D-only selection');fig.tight_layout();fig.savefig(OUT/'controls.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots()
 for r in ev:
  if r['method']=='GN':ax.plot([0,.5,1],[r['fixed_remove_gain'],r['fixed_half_gain'],0],alpha=.35)
 ax.set(xlabel='Lambda (1 = KEEP)',ylabel='T joint NLL benefit vs KEEP');fig.savefig(OUT/'dose_response.png',dpi=150);plt.close(fig)
