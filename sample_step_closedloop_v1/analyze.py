import statistics as st,itertools
import numpy as np
from .common import *

def summarize(trials):
 cells=[];seeds=[];pairs=[]
 for seed in [45,46,47]:
  for task in [6,7,8]:
   arrays={a:[r['success'] for r in sorted([r for r in trials if r['seed']==seed and r['task']==task and r['arm']==a],key=lambda r:r['trial'])] for a in ARMS}
   assert all(len(v)==50 for v in arrays.values())
   for a,arr in arrays.items():cells.append(dict(seed=seed,task=task,arm=a,successes=sum(arr),trials=50,success_rate=2*sum(arr)))
   for baseline in ['MLR','TASK_STEP_QP']:
    a=arrays[baseline];b=arrays['SAMPLE_STEP_QP'];pairs.append(dict(seed=seed,task=task,baseline=baseline,delta_pp=2*(sum(b)-sum(a)),wins=sum(y>x for x,y in zip(a,b)),losses=sum(y<x for x,y in zip(a,b)),ties=sum(y==x for x,y in zip(a,b))))
  for arm in ARMS:
   rs=[r for r in cells if r['seed']==seed and r['arm']==arm]
   seeds.append(dict(seed=seed,arm=arm,seen=st.mean(r['success_rate'] for r in rs),old=st.mean(r['success_rate'] for r in rs if r['task']<8),new=next(r['success_rate'] for r in rs if r['task']==8),worst_old=min(r['success_rate'] for r in rs if r['task']<8)))
 pooled=[dict(arm=a,**{k:st.mean(r[k] for r in seeds if r['arm']==a) for k in ['seen','old','new','worst_old']}) for a in ARMS]
 differences=[]
 for baseline in ['MLR','TASK_STEP_QP']:
  for seed in [45,46,47,'ALL']:
   source=pooled if seed=='ALL' else [r for r in seeds if r['seed']==seed]
   b=next(r for r in source if r['arm']==baseline);s=next(r for r in source if r['arm']=='SAMPLE_STEP_QP')
   differences.append(dict(seed=seed,baseline=baseline,**{k:s[k]-b[k] for k in ['seen','old','new','worst_old']}))
 return cells,seeds,pooled,pairs,differences

def decision(differences,cells):
 pooled={r['baseline']:r for r in differences if r['seed']=='ALL'};mlr=pooled['MLR'];individual=[r for r in differences if r['seed']!='ALL'];dm=[r for r in individual if r['baseline']=='MLR']
 severe_tasks=[]
 for task in [6,7,8]:
  bad=0
  for seed in [45,46,47]:
   v={r['arm']:r['success_rate'] for r in cells if r['seed']==seed and r['task']==task};bad+=v['SAMPLE_STEP_QP']-v['MLR']<=-10
  if bad>=2:severe_tasks.append(task)
 checks=dict(both_primary_ge2=all(r['seen']>=2 for r in pooled.values()),both_majority_positive=all(sum(r['seen']>0 for r in individual if r['baseline']==b)>=2 for b in pooled),old_positive=mlr['old']>0,new_mean_within_budget=mlr['new']>=-2,no_large_seed_primary_drop=all(r['seen']>-5 for r in dm),no_large_seed_new_drop=all(r['new']>-5 for r in dm),no_repeated_task_drop=not severe_tasks)
 status='CANDIDATE_FOR_LARGER_STUDY' if all(checks.values()) else ('NOT_SUPPORTED' if mlr['seen']<=0 or mlr['new']<=-5 else 'INCONCLUSIVE')
 return dict(status=status,checks=checks,severe_tasks=severe_tasks,scope='Development closed-loop screening only, not independent confirmation or full lifelong evidence.')

def run():
 verify();p=read(OUT/'PROTOCOL_FROZEN.json');trials=[]
 for c in p['cells']:
  d=OUT/'evaluation'/c['point']/f"task{c['task']}"/c['arm'];meta=read(d/'COMPLETE.json')
  for n,h in meta['files'].items():assert sha(d/n)==h
  trials+=read(d/'TRIALS.json')
 assert len(trials)==1350
 cells,seeds,pooled,pairs,diffs=summarize(trials)
 rng=np.random.RandomState(20260916);boot={b:[] for b in ['MLR','TASK_STEP_QP']}
 arr={(s,t,a):np.asarray([r['success'] for r in sorted([x for x in trials if x['seed']==s and x['task']==t and x['arm']==a],key=lambda x:x['trial'])]) for s,t,a in itertools.product([45,46,47],[6,7,8],ARMS)}
 for _ in range(10000):
  # One resampled initial-state index list/task, linked across seeds and arms.
  ix={t:rng.randint(0,50,50) for t in [6,7,8]}
  for b in boot:boot[b].append(float(np.mean([np.mean(arr[s,t,'SAMPLE_STEP_QP'][ix[t]]-arr[s,t,b][ix[t]]) for s,t in itertools.product([45,46,47],[6,7,8])]))*100)
 intervals={b:dict(lower=float(np.percentile(v,2.5)),upper=float(np.percentile(v,97.5))) for b,v in boot.items()}
 for n,r in [('trials',trials),('cells',cells),('seeds',seeds),('pooled',pooled),('paired_trials',pairs),('contrasts',diffs)]:csvout(OUT/(n+'.csv'),r)
 events=[]
 for name in p['points']:
  for a in ARMS:
   rs=rows(OUT/'training'/name/a/'events.jsonl');assert len(rs)==300
   events.append(dict(point=name,arm=a,steps=300,update_seconds=sum(r['seconds'] for r in rs),triggers=sum(r.get('trigger',False) for r in rs),mean_actual_norm_ratio=st.mean(r.get('actual_norm_ratio',1) for r in rs),total_actual_reference_violations=sum(r.get('actual_positive',0) for r in rs)))
 csvout(OUT/'training_cost.csv',events)
 write(OUT/'SUMMARY.json',dict(status='COMPLETE',trials=1350,seeds=seeds,pooled=pooled,contrasts=diffs,conditional_paired_initial_state_intervals=intervals,interval_warning='Conditional on these3trained development seeds; not training-seed uncertainty or independent confirmatory significance.'))
 d=decision(diffs,cells);write(OUT/'DECISION.json',d)
 md=['# 三seed逐样本位移QP闭环筛查','', '三个开发seed；每版本task8续训300步，固定终点评估task6/7/8各50trial，总1350trial。成功率与差值单位为百分点。', '', '|seed|方法|三任务等权成功率|旧任务6/7|新任务8|','|---|---|---:|---:|---:|']
 for r in seeds+[dict(seed='ALL',**r) for r in pooled]:md.append('|%s|%s|%.3f|%.3f|%.3f|'%(r['seed'],r['arm'],r['seen'],r['old'],r['new']))
 md+=['','预登记判断：'+d['status'],'','```json',json.dumps(d,indent=2),'```','','原始trial、逐seed差值和训练成本均随报告提供。独立复核应为PASS后使用结果。','固定参考窗口被每步重复使用；本轮只覆盖task8的300步，不代表完整终身训练。三个seed都是开发seed，区间仅描述固定模型下配对初始状态的不确定性。没有根据中间成功率选点、调参、挑checkpoint或追加trial。']
 (OUT/'FINAL_REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('ANALYSIS_COMPLETE',d['status'],flush=True)
if __name__=='__main__':run()
