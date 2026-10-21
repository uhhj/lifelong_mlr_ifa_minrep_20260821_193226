"""Read-only exact decomposition of accepted CAET OBJECT outcomes. No new inference."""
import json,hashlib,statistics
from fractions import Fraction as F
from pathlib import Path
from audit_rctr_completion_caet_progress import metrics
R=Path(__file__).resolve().parent
P=R/'caet_object_analysis_20261021'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(v):return dict(fraction=str(v),pp=float(100*v))
def main():
 s=json.loads((R/'caet_object_three_seed_final_20261021/SUMMARY.json').read_text())
 rows=[];cells={};cost=[]
 for e in s['evidence']:
  ap,dp=R/e['audit'],R/e['details'];assert sha(ap)==e['audit_sha256'] and sha(dp)==e['details_sha256']
  audit=json.loads(ap.read_text());c=json.loads(dp.read_text())['caet'];seed=e['seed']
  new,m,ni=metrics(c['formal']['trials'],seed);old,b,oi=metrics(c['completion']['baseline_trials'],seed)
  assert all(F(audit['metrics'][k]['fraction'])==v for k,v in m.items())
  task_auc={t:sum(new[st,t]-old[st,t] for st in range(t,10))/F(10-t)/4 for t in range(6,10)}
  assert sum(task_auc.values())==m['AUC']-b['AUC']
  paired={}
  for st,t in new:
   delta=new[st,t]-old[st,t];cells.setdefault((st,t),[]).append((old[st,t],new[st,t]))
   records=[(oi[st,t,i]['success'],ni[st,t,i]['success']) for i in range(20)]
   paired[f'{st}/{t}']=dict(FS=sum(not x and y for x,y in records),SF=sum(x and not y for x,y in records),delta=pack(delta))
  rows.append(dict(seed=seed,auc_delta=pack(m['AUC']-b['AUC']),
      auc_contribution_by_task={str(k):pack(v) for k,v in task_auc.items()},outcome_changes=paired))
  preds=audit['effect_audit']['predictors']
  cost.append(dict(seed=seed,main_epoch_seconds=audit['summed_epoch_seconds'],
      active_auxiliary=sum(v['active_updates'] for v in audit['auxiliary'].values()),
      effect_updates=sum(p['optimizer_steps'] for p in preds),
      predictors=[{k:p[k] for k in ('task','observed_training_positive_pairs','heldout_final_normalized_mae','action_gradient_l1')} for p in preds]))
 matrix=[dict(stage=st,task=t,ra=pack(sum(x for x,y in v)/3),caet=pack(sum(y for x,y in v)/3),
              delta=pack(sum(y-x for x,y in v)/3)) for (st,t),v in cells.items()]
 contributions={str(t):pack(sum(F(r['auc_contribution_by_task'][str(t)]['fraction']) for r in rows)/3) for t in range(6,10)}
 mean=sum(F(r['auc_delta']['fraction']) for r in rows)/3
 without42=sum(F(r['auc_delta']['fraction']) for r in rows if r['seed']!=42)/2
 assert mean==F(s['summary']['paired_delta']['AUC']['mean_fraction'])
 out=dict(status='READ_ONLY_ANALYSIS_PASS',rows=rows,mean_matrix=matrix,auc_contributions=contributions,
    mean_auc_delta=pack(mean),descriptive_mean_excluding_selection_seed42=pack(without42),
    cost=cost,evidence=s['evidence'],training_calls=0,environment_calls=0,
    caution='FS/SF are same-initial outcome changes, not causal rescue counts: global action noise is not trajectory-isolated.')
 P.mkdir(exist_ok=True);(P/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
 report='''# CAET 在 OBJECT 的三 seed 结果分析

基于已独立验收的 seed41/42/43 全部600条正式记录与对应RA基线，重新读取原episode复算。没有新增训练、模型推理或环境评估。

结论：CAET有正的平均收益，但不是稳定的正向迁移；总AUC净收益主要由Task9贡献，另有部分后续阶段的旧任务改善。seed42贡献集中，seed41明显退化，不能凭通过扩展门槛就认定方法稳定有效。

|指标（%）|RA均值|CAET均值|配对差（百分点）|
|---|---:|---:|---:|
'''
 for k in s['summary']['RA']:
  report+=f"|{k}|{s['summary']['RA'][k]['mean_percent']:.6f}|{s['summary']['CAET'][k]['mean_percent']:.6f}|{s['summary']['paired_delta'][k]['mean_percent']:+.6f}|\n"
 report+='''
AUC逐seed差：41 −1.770833、42 +5.520833、43 +0.833333pp。平均+1.527778，配对差样本SD=3.695104pp；只有三个seed且是在看到42改善后扩展，不作显著性或稳健性结论。仅作敏感性描述，41/43平均差为−0.46875pp；这不替代已约定三seed主结果，也不是独立预注册验证。

## 指标必须按实际代码解释

AUC是四个任务各自“从学到结束”的成功率平均，再对任务等权平均；不是全部200条episode简单平均。Task6每个cell权重1/16、Task7每cell1/12、Task8每cell1/8、Task9单cell1/4。

本项目FWT是四个对角cell（各任务训练100epoch后）的平均成功率，不是未训练新任务的零样本迁移，也没有衡量学习速度。NBT是首次学完成功率减去后续阶段成功率，先按每个旧任务平均，再对旧任务平均；因此越低代表这个定义下遗忘越少，负数表示后续平均超过初学时。CAET的NBT下降5pp是这个定义下的有利方向，但其中3.888889pp可由旧任务初学基准变低解释，只有另外1.111111pp对应按NBT权重汇总的后续旧任务绝对表现提高。这是代数分解而非因果归因，不能把5pp都宣传为抗遗忘收益。

## 全部阶段平均差

|完成训练阶段|评估任务|RA均值%|CAET均值%|差pp|
|---|---|---:|---:|---:|
'''
 for v in matrix:report+=f"|{v['stage']}|{v['task']}|{v['ra']['pp']:.3f}|{v['caet']['pp']:.3f}|{v['delta']['pp']:+.3f}|\n"
 report+='\n各任务对总AUC差的贡献（pp）：'+', '.join(f"Task{t} {v['pp']:+.6f}" for t,v in contributions.items())+'.\n'
 report+='''
Task7最需要警惕：初学时平均下降10pp，在Task8之后仍下降5pp，最终下降1.667pp；seed41的Task7三个cell分别−20/−20/−15pp。Task8初学略降1.667pp，最终却升3.333pp。Task9初学平均升8.333pp，但41/42为+20/+15、43为−10，尚不稳定。

Task9贡献AUC +2.083333pp，比总净增+1.527778pp还大；Task6–8合计为−0.555556pp。因此总体AUC提高不能概括成“全面保住了旧知识”。

seed42的主要改善来自最终Task6由10%到55%（+45pp），以及Task9的+15pp；seed41最终Task6/7分别−25/−15pp，Task9+20pp仍无法抵消。seed43的收益则更多来自Task7完成后的旧Task6 +30pp，最终Task9反而−10pp。三seed不是同一种收益模式。

## 机制证据能说明什么

所有三seed Task6模型实际张量同父基线；四阶段RA成员、载荷、IFA规则与初始化绑定核验通过。Task7–9辅助损失每seed真实激活约89792次，另各1240次效果预测器更新，有非零有限动作梯度。因此不是RCTR那样“辅助监督从未启用”。

代码中的训练目标是BC + 0.1 IFA + 0.05效果误差。预测器先在当前任务和真实旧记忆上拟合后冻结；策略把当前选中GMM分量采样的动作送入预测器，与专家物理效果标签做SmoothL1。有效标签覆盖当前和旧样本，不只训练当前任务；所以它可以改变旧任务回放时的动作梯度，观察到保持/后向改善是合理的，但不能由此证明是哪一条机制造成收益。

预测器通过非零动作梯度检查只说明“能反传”，不保证策略动作离开专家分布时预测准确，更不保证梯度方向改善真实控制。原训练/留出误差是专家动作数据上的预测证据，不是任意候选动作的反事实因果验证。BC与效果误差的梯度可能冲突、当前任务与旧任务分布可能不匹配，这是Task7退化及seed差异的合理解释，但目前没有梯度夹角/策略动作域外误差证据，不能定论。

另一边界是使用argmax分量：辅助路径对选中均值与尺度可微，分量选择detach，未直接优化离散模式切换。专家相似效果和最终任务成功也不是同一个目标。因此“学习了效果规律”并不必然导致FWT提高。

## 判断与后续优先级

保留为有潜力但不稳定的结果，不建议据此宣称已证实正向迁移。优先解释Task7为何普遍受损、seed41旧Task6/7为何退化，以及seed42最终Task6的+45pp是否体现可复现保持；不要只围绕表现最好的seed42叙事。现有结果支持继续提出机制问题，不支持无条件迁移到GOAL，后者还存在已查明的支持不足。

本轮只分析既有资产，没有启动新消融或改变损失/阈值。三seed样本量小、每cell20条、全局动作RNG未按轨迹隔离；同初态的FS/SF变化不能直接称为动作修复因果效果。效果预测额外拟合成本、pretrain state元数据未知与调度补全继续披露。
'''
 (P/'REPORT.md').write_text(report,encoding='utf8');print(json.dumps(dict(status=out['status'],auc_contributions=contributions,without42=pack(without42)),indent=2))
if __name__=='__main__':main()
