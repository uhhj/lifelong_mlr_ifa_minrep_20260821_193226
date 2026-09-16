import statistics as st,itertools
import numpy as np
from .common import *

def summarize(trials):
 cells=[];seeds=[];pairs=[]
 for seed in [45]:
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
  for seed in [45,'ALL']:
   source=pooled if seed=='ALL' else [r for r in seeds if r['seed']==seed]
   b=next(r for r in source if r['arm']==baseline);s=next(r for r in source if r['arm']=='SAMPLE_STEP_QP')
   differences.append(dict(seed=seed,baseline=baseline,**{k:s[k]-b[k] for k in ['seen','old','new','worst_old']}))
 return cells,seeds,pooled,pairs,differences

def decision(differences,cells):
 return dict(status='SINGLE_SEED_DESCRIPTIVE_ONLY', deferred_seeds=[46,47], scope='User requested seed45 first; original one-seed decision criteria cannot be applied. No automatic expansion based on this result.')


def run():
 verify();p=read(OUT/'PROTOCOL_FROZEN.json');p['cells']=[c for c in p['cells'] if c['environment_seed']==45];p['points']=p['points'][:1];trials=[]
 for c in p['cells']:
  d=OUT/'evaluation'/c['point']/f"task{c['task']}"/c['arm'];meta=read(d/'COMPLETE.json')
  for n,h in meta['files'].items():assert sha(d/n)==h
  trials+=read(d/'TRIALS.json')
 assert len(trials)==450
 cells,seeds,pooled,pairs,diffs=summarize(trials)
 rng=np.random.RandomState(20260916);boot={b:[] for b in ['MLR','TASK_STEP_QP']}
 arr={(s,t,a):np.asarray([r['success'] for r in sorted([x for x in trials if x['seed']==s and x['task']==t and x['arm']==a],key=lambda x:x['trial'])]) for s,t,a in itertools.product([45],[6,7,8],ARMS)}
 for _ in range(10000):
  # One resampled initial-state index list/task, linked across seeds and arms.
  ix={t:rng.randint(0,50,50) for t in [6,7,8]}
  for b in boot:boot[b].append(float(np.mean([np.mean(arr[s,t,'SAMPLE_STEP_QP'][ix[t]]-arr[s,t,b][ix[t]]) for s,t in itertools.product([45],[6,7,8])]))*100)
 intervals={b:dict(lower=float(np.percentile(v,2.5)),upper=float(np.percentile(v,97.5))) for b,v in boot.items()}
 for n,r in [('trials',trials),('cells',cells),('seeds',seeds),('pooled',pooled),('paired_trials',pairs),('contrasts',diffs)]:csvout(OUT/('seed45_'+n+'.csv'),r)
 events=[]
 for name in p['points']:
  for a in ARMS:
   rs=rows(OUT/'training'/name/a/'events.jsonl');assert len(rs)==300
   events.append(dict(point=name,arm=a,steps=300,update_seconds=sum(r['seconds'] for r in rs),triggers=sum(r.get('trigger',False) for r in rs),mean_actual_norm_ratio=st.mean(r.get('actual_norm_ratio',1) for r in rs),total_actual_reference_violations=sum(r.get('actual_positive',0) for r in rs)))
 csvout(OUT/'seed45_training_cost.csv',events)
 write(OUT/'SUMMARY_SEED45.json',dict(status='COMPLETE',trials=450,seeds=seeds,pooled=pooled,contrasts=diffs,conditional_paired_initial_state_intervals=intervals,interval_warning='Conditional on this single trained development seed; not training-seed uncertainty or independent confirmatory significance.'))
 d=decision(diffs,cells);write(OUT/'DECISION_SEED45.json',d)
 md=['# seed45逐样本位移QP闭环初筛','', '一个开发seed45；每版本task8续训300步，固定终点评估task6/7/8各50trial，总450trial。成功率与差值单位为百分点。', '', '|seed|方法|三任务等权成功率|旧任务6/7|新任务8|','|---|---|---:|---:|---:|']
 for r in seeds+[dict(seed='ALL',**r) for r in pooled]:md.append('|%s|%s|%.3f|%.3f|%.3f|'%(r['seed'],r['arm'],r['seen'],r['old'],r['new']))
 md+=['','预登记判断：'+d['status'],'','```json',json.dumps(d,indent=2),'```','','原始trial、配对差值和训练成本均随报告提供。独立复核应为PASS后使用结果。','固定参考窗口被每步重复使用；本轮只覆盖task8的300步，不代表完整终身训练。seed45是开发seed，区间仅描述固定模型下配对初始状态的不确定性。没有根据中间成功率选点、调参、挑checkpoint或追加trial。']
 (OUT/'FINAL_REPORT_SEED45.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('ANALYSIS_COMPLETE',d['status'],flush=True)
if __name__=='__main__':run()
