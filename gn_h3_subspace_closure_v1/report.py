import csv,json,numpy as np
from scipy.stats import spearmanr,pearsonr
from .common import *
def loadrows(name):return list(csv.DictReader((OUT/name).open()))
def corr(a,b,rank=True):
 if np.std(a)==0 or np.std(b)==0:return None
 return float((spearmanr if rank else pearsonr)(a,b)[0])
def fmt(v):return 'NA' if v is None else f'{v:.5g}'
def main():
 p=read(OUT/'PROTOCOL.json');a=read(OUT/'A_REPORT.json');ar=loadrows('A_DIRECTIONAL_CLOSURE.csv');byh=[]
 for factor in p['A']['h_factors']:
  rs=[r for r in ar if float(r['h_factor'])==factor];errors=[float(r['relative_error']) for r in rs]
  byh.append(dict(h_factor=factor,h=p['epsilon0']*factor,closed_count=sum(r['closed']=='True' for r in rs),n=len(rs),median_relative_error=float(np.median(errors)),maximum_relative_error=float(max(errors))))
 table('A_CLOSURE_BY_SCALE.csv',byh)
 summary=dict(A_status=a['status'],A_passed=a['passed_groups'],A_total=a['total_groups'],A_common_closed_factors=[r['h_factor'] for r in byh if r['closed_count']==r['n']],
  training_active=False,formal_training_resumed=False,formal_solver_modified=False,optimizer_steps_executed=0)
 text=f'''# 方向导数闭合与基不变子空间H3

本轮只包含A、B两个实验。同一seed45原生MLR task8 ep99、同一task8/task9样本与专家标签；策略头float64、冻结解码器float32。未执行正式训练或优化器更新。

## A：真实GMM NLL方向导数闭合

结果：**{a['status']}，{a['passed_groups']}/{a['total_groups']}组通过**。包括mean-GN前16方向、既有Fisher前16方向及旧评价梯度/新任务梯度/随机方向；分别在旧、新评价库验证。每组必须在事前固定网格的两个相邻小步长上满足相对误差≤1e-4。

中央差分按逐窗口NLL之差再平均计算，避免先减两个大均值增加舍入抵消。误差分母为max(|gᵀv|,1e-8||g||)，近零导数单列，不将其解释为安全方向。

| h/epsilon0 | 闭合组数 | 相对误差中位数 | 最大相对误差 |
|---|---:|---:|---:|
'''
 for r in byh:text+=f"| {r['h_factor']:.1g} | {r['closed_count']}/{r['n']} | {r['median_relative_error']:.6g} | {r['maximum_relative_error']:.6g} |\n"
 text+='\n这项检查验证当前自动微分/实际损失接口的局部一致性，不等于所有有限更新已位于线性区间。该步长网格只作数值闭合，不用于按结果调整B的步长。\n'
 if a['status']!='PASS':
  summary['B_status']='NOT_RUN_A_FAILED';write(OUT/'SUMMARY.json',summary);(OUT/'FINAL_REPORT.md').write_text(text+'\nA未通过，因此停止，未运行B。\n');return
 data=loadrows('B_SUBSPACE_EFFECTS.csv');metrics=[];rank_rows=[];tol=p['B']['loss_effect_tolerance']
 for r in data:
  po=float(r['pred_old_cross']);pn=float(r['pred_new_cross']);old=float(r['measured_old']);new=float(r['measured_new']);weak=r['weak_projection']=='True' or abs(po)<=tol
  label='WEAK' if weak else ('COMPATIBLE' if po<0 else 'CONFLICT')
  correct=not weak and abs(old)>tol and np.sign(po)==np.sign(old)
  joint=(old<=tol and new<=tol) if label=='COMPATIBLE' else ((old>tol and new<-tol) if label=='CONFLICT' else False)
  rank_rows.append(dict(geometry=r['geometry'],rank=int(r['rank']),scale=float(r['scale']),label=label,sign_correct=bool(correct),joint_behavior=bool(joint),**{k:float(r[k]) for k in ['score_old_relation','projected_new_gradient_norm','projection_energy_fraction','perturbation_norm','pred_old_cross','pred_old_same_probe','pred_new_cross','pred_new_same_probe','measured_old','measured_new']}))
 for geometry in p['B']['geometries']:
  for scale in p['B']['scales']:
   rs=[r for r in rank_rows if r['geometry']==geometry and r['scale']==scale];true=np.array([r['measured_old'] for r in rs]);cross=np.array([r['pred_old_cross'] for r in rs]);same=np.array([r['pred_old_same_probe'] for r in rs])
   metrics.append(dict(geometry=geometry,scale=scale,n=4,cross_sign_correct=sum(r['sign_correct'] for r in rs),joint_correct=sum(r['joint_behavior'] for r in rs),weak=sum(r['label']=='WEAK' for r in rs),
    same_probe_sign_correct=int(np.sum((abs(same)>tol)&(abs(true)>tol)&(np.sign(same)==np.sign(true)))),cross_spearman=corr(cross,true),same_probe_spearman=corr(same,true),
    cross_pearson=corr(cross,true,False),same_probe_pearson=corr(same,true,False),same_probe_relative_linearization_error=float(np.linalg.norm(true-same)/max(np.linalg.norm(same),1e-30)),
    mean_old_change=float(true.mean()),mean_new_change=float(np.mean([r['measured_new'] for r in rs]))))
 table('B_RANK_RESULTS.csv',rank_rows);table('B_METRICS.csv',metrics)
 comparisons=[]
 for geometry in ['MEAN_GN','MODEL_FISHER_MC8']:
  rows=[]
  for scale in p['B']['scales']:
   x=next(r for r in metrics if r['geometry']==geometry and r['scale']==scale);y=next(r for r in metrics if r['geometry']=='RANDOM' and r['scale']==scale)
   rows.append(dict(scale=scale,sign_count_difference=x['cross_sign_correct']-y['cross_sign_correct'],joint_count_difference=x['joint_correct']-y['joint_correct']))
  comparisons.append(dict(geometry=geometry,by_scale=rows,dominates_random_on_descriptive_counts=all(r['sign_count_difference']>=0 and r['joint_count_difference']>=0 for r in rows) and any(r['sign_count_difference']>0 or r['joint_count_difference']>0 for r in rows)))
 advantage=any(r['dominates_random_on_descriptive_counts'] for r in comparisons)
 summary.update(B_status='COMPLETE',B_basis_invariance=read(OUT/'BASIS_INVARIANCE.json')['status'],B_metrics=metrics,B_comparisons=comparisons,
  conclusion='DESCRIPTIVE_ADVANTAGE_REQUIRES_CONFIRMATION' if advantage else 'NO_ESTABLISHED_GEOMETRY_ADVANTAGE_ON_THIS_PAIR',
  inference_limit='Only one task pair, one random basis, four nested ranks; descriptive counts are not independent experiments or proof of method superiority.')
 infinitesimal=[]
 for geometry in p['B']['geometries']:
  rr=[r for r in rank_rows if r['geometry']==geometry and r['scale']==1.]
  correct=sum(abs(r['pred_old_cross'])>tol and abs(r['pred_old_same_probe'])>tol and np.sign(r['pred_old_cross'])==np.sign(r['pred_old_same_probe']) for r in rr)
  joint=sum(r['pred_old_same_probe']<=tol and r['pred_new_same_probe']<=tol for r in rr if r['label']=='COMPATIBLE')
  infinitesimal.append(dict(geometry=geometry,old_cross_to_probe_sign_agreement=int(correct),n=4,predicted_compatible=sum(r['label']=='COMPATIBLE' for r in rr),predicted_conflict=sum(r['label']=='CONFLICT' for r in rr),probe_infinitesimal_joint_compatible=int(joint)))
 summary['B_infinitesimal_relation_descriptive']=infinitesimal
 summary['postmeasurement_decomposition']='Derived only from existing score_old_relation/score_old_probe and probe derivative columns; no extra model evaluation, no revised finite-step decision.'
 write(OUT/'SUMMARY.json',summary)
 text+='''
## B：子空间整体关系

关系量为R=g_oldᵀ P_S g_new；P_S=U(UᵀU)^(-1)Uᵀ，显式修正保存基的微小非正交误差。

实际扰动为delta=-epsilon P_S g_new/||P_S g_new||，因此一阶预测是-epsilon R/||P_S g_new||，不能直接把R当作损失变化。三种空间在每档具有相同参数扰动范数。原始投影能量和关系量同时保留，未把随机空间较小的投影量隐去。

Mean-GN复用已有top32中的相应前缀。Fisher仅在同一1024个固定模型样本上补足top21，保留原16列，只保存新增5列。随机空间为一次固定21维正交基，各rank使用嵌套前缀。不同rank并非独立重复。

主关系梯度来自old_gradient/new_gradient；测量来自轨迹分离的old_probe/new_probe。评价库自身梯度仅用于局部线性校准。

| 几何 | 尺度 | 跨轨迹符号正确 | 联合行为正确 | 同库校准符号正确 | 同库相对线性化误差 |
|---|---:|---:|---:|---:|---:|
'''
 for r in metrics:text+=f"| {r['geometry']} | {r['scale']} | {r['cross_sign_correct']}/4 | {r['joint_correct']}/4 | {r['same_probe_sign_correct']}/4 | {r['same_probe_relative_linearization_error']:.5g} |\n"
 text+='\n联合行为：预测相容时要求新旧损失均不恶化；预测冲突时要求旧损失增加、新损失下降。弱或中性效应保留在保守分母。\n\n'
 for scale in p['B']['scales']:
  text+=f'## 逐rank结果：epsilon/epsilon0={scale}\n\n| 几何 | r | 预测类别 | 预测旧ΔL | 实测旧ΔL | 实测新ΔL | 投影能量占比 |\n|---|---:|---|---:|---:|---:|---:|\n'
  for r in rank_rows:
   if r['scale']==scale:text+=f"| {r['geometry']} | {r['rank']} | {r['label']} | {r['pred_old_cross']:.6g} | {r['measured_old']:.6g} | {r['measured_new']:.6g} | {r['projection_energy_fraction']:.6g} |\n"
 text+='\n## 尺度无关的一阶关系分解（已有投影量的描述性重算）\n\n| 几何 | 旧任务跨轨迹与评价梯度符号一致 | 预测相容/冲突数 | 评价梯度同时支持新旧改善的相容项 |\n|---|---:|---:|---:|\n'
 for r in infinitesimal:text+=f"| {r['geometry']} | {r['old_cross_to_probe_sign_agreement']}/4 | {r['predicted_compatible']}/{r['predicted_conflict']} | {r['probe_infinitesimal_joint_compatible']}/{r['predicted_compatible']} |\n"
 text+='\n这里比较两个轨迹组的解析一阶关系，消除了有限位移的曲率余项；它不是额外扰动实验，也不是用评价数据选择方向。Fisher旧任务符号一致4/4是局部正面信号，但四项全为相容，不能证明对共享与冲突两类都能辨别。新任务跨轨迹泛化仍需分别检查。\n'
 text+=f'''\n## 结论与边界

描述性结论：**{summary['conclusion']}**。判断依据为固定两档尺度的符号正确数和新旧联合行为数相对random的比较；这不是显著性检验或正式方法推进门。完整连续指标见B_METRICS.csv，不以计数掩盖损失量级。

子空间内正交换基检查：{summary['B_basis_invariance']}。这一不变性针对固定span；不能消除更改rank、参考数据或接近简并特征值的截断所带来的子空间变化。

A若通过，支持gᵀv确为当前GMM NLL的方向导数。B没有优于random时，只能说本例下功能几何对关系判别的额外价值未建立，不能推出功能方向不存在、梯度公式错误或所有GN/Fisher方法无效。

本轮没有执行真实AdamW位移投影、训练更新、逐样本QP或闭环。Fisher仍是固定模型score积分估计，不是专家NLL Hessian。

`training_active=false`；`formal_training_resumed=false`。
'''
 (OUT/'FINAL_REPORT.md').write_text(text)
 write(OUT/'STATUS.json',dict(stage='COMPLETE',A_status=a['status'],B_status='COMPLETE',training_active=False,formal_training_resumed=False));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
