import csv,numpy as np
from scipy.stats import pearsonr,spearmanr
from .common import *
def correlation(a,b,spearman=True):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 return float((spearmanr if spearman else pearsonr)(a,b)[0])
def main():
 p=read(OUT/'PROTOCOL.json');data=list(csv.DictReader((OUT/'DIRECTION_EFFECTS.csv').open()));summaries=[]
 for kind in ['MEAN_GN','MODEL_FISHER_MC8']:
  for scale in p['scales']:
   rs=[r for r in data if r['geometry']==kind and float(r['scale'])==scale]
   for task in ['old','new']:
    for mode in ['same_probe','cross_trajectory']:
     pred=np.array([float(r['pred_'+task+'_'+mode]) for r in rs]);meas=np.array([float(r['measured_'+task]) for r in rs]);tol=p['loss_effect_tolerance']
     eligible=(abs(pred)>tol)&(abs(meas)>tol)&np.array([r['weak_new_projection']=='False' for r in rs]);correct=eligible&(np.sign(pred)==np.sign(meas))
     delta=meas-pred
     summary=dict(geometry=kind,scale=scale,task=task,predictor=mode,n=len(rs),evaluable=int(eligible.sum()),correct=int(correct.sum()),conservative_sign_accuracy=float(correct.mean()),conditional_sign_accuracy=float(correct.sum()/eligible.sum()) if eligible.any() else None,
      pearson=correlation(pred,meas,False),spearman=correlation(pred,meas),relative_linearization_error=float(np.linalg.norm(delta)/max(np.linalg.norm(pred),1e-30)),absolute_error_norm=float(np.linalg.norm(delta)),
      mean_abs_pred=float(abs(pred).mean()),mean_abs_measured=float(abs(meas).mean()),max_abs_error=float(abs(delta).max()),mean_measured=float(meas.mean()))
     summary['good_local_calibration']=bool(summary['evaluable']==16 and summary['conservative_sign_accuracy']>=.9375 and summary['pearson'] is not None and summary['pearson']>=.9 and summary['spearman']>=.9)
     summaries.append(summary)
 table('METRICS.csv',summaries)
 primary={kind:[r for r in summaries if r['geometry']==kind and r['task']=='old' and r['predictor']=='same_probe'] for kind in ['MEAN_GN','MODEL_FISHER_MC8']}
 evidence={}
 for kind,rs in primary.items():
  first,last=rs[0],rs[-1];factor=first['relative_linearization_error']/max(last['relative_linearization_error'],1e-30)
  slope=float(np.polyfit(np.log([r['scale'] for r in rs]),np.log([max(r['absolute_error_norm'],1e-300) for r in rs]),1)[0])
  evidence[kind]=dict(finite_step_effect_supported=bool(last['good_local_calibration'] and last['conservative_sign_accuracy']-first['conservative_sign_accuracy']>=.125 and factor>=4),
   relative_error_reduction=factor,absolute_error_loglog_slope=slope,good_scales=[r['scale'] for r in rs if r['good_local_calibration']],smallest_scale_good=last['good_local_calibration'])
 advantages=[]
 for mean,fish in zip(primary['MEAN_GN'],primary['MODEL_FISHER_MC8']):
  da=fish['conservative_sign_accuracy']-mean['conservative_sign_accuracy'];dr=fish['spearman']-mean['spearman'] if fish['spearman'] is not None and mean['spearman'] is not None else None
  advantages.append(dict(scale=mean['scale'],accuracy_difference=da,spearman_difference=dr,evaluable_difference=fish['evaluable']-mean['evaluable'],material_advantage=bool(da>=.125 and dr is not None and dr>=.15 and fish['evaluable']>=mean['evaluable'])))
 geometry_adv=sum(r['material_advantage'] for r in advantages)>=2
 both_good=all(r['smallest_scale_good'] for r in evidence.values());only_small=both_good and all(max(r['good_scales'])<=.0625 for r in evidence.values())
 if only_small:classification='BOTH_GEOMETRIES_LOCAL_ONLY'
 elif both_good:classification='LOCAL_CALIBRATION_RECOVERED'
 elif geometry_adv:classification='GEOMETRY_DEPENDENCE_OBSERVED'
 else:classification='MIXED_OR_UNRESOLVED_LOCAL_CALIBRATION'
 summary=dict(classification=classification,A_finite_step_evidence=evidence,B_fisher_advantage_established_on_fixed_criteria=geometry_adv,geometry_advantages=advantages,primary=primary,
  calibration_scope='same-probe directional-derivative check, not independent relationship prediction',
  training_active=False,formal_training_resumed=False,formal_solver_modified=False,optimizer_steps_executed=0)
 write(OUT/'SUMMARY.json',summary)
 eff=read(OUT/'EFFICIENCY.json');text='''# H3步长×功能几何诊断

本轮是同一checkpoint、同一数据的2×4局部诊断。复用原mean-GN前16方向；另一组为完整GMM模型Fisher的前16方向。正式模型、buffer、损失、优化器与solver均不修改。

主预测使用**评价样本自身梯度**，检验一阶近似的有限步长误差；跨轨迹关系预测单独报告。相同样本校准良好不能当作关系泛化通过。

## 主结果：旧任务同评价样本梯度

| 几何 | 步长/epsilon0 | 符号正确 | 可分辨方向 | Pearson | Spearman | 相对线性化误差 |
|---|---:|---:|---:|---:|---:|---:|
'''
 for kind,rs in primary.items():
  for r in rs:text+=f"| {kind} | {r['scale']} | {r['correct']}/16 | {r['evaluable']}/16 | {r['pearson']:.5f} | {r['spearman']:.5f} | {r['relative_linearization_error']:.6g} |\n"
 text+='\n相对误差为||实测-一阶预测||₂/||一阶预测||₂；微小/中性效应保留在保守准确率分母。epsilon0沿用上一轮约0.0128804，不重新估计或依据结果改变。每个方向在同尺度具有相同参数扰动范数。\n'
 text+='\n## A：有限步长证据\n\n'
 for kind,r in evidence.items():text+=f"- {kind}：预登记局部恢复判据={r['finite_step_effect_supported']}；相对误差缩小{r['relative_error_reduction']:.3g}倍；绝对余项对步长log-log斜率={r['absolute_error_loglog_slope']:.4f}；达到校准门的尺度={r['good_scales']}。\n"
 text+='\n误差随缩步减小说明有限扰动的一阶近似问题，但这里没有分离光滑曲率、ReLU分段变化等来源。即使最小步长仍未过门，也不能否定梯度方向导数公式，或直接判定逐方向关系在所有尺度无效。\n'
 text+=f'\n## B：功能度量比较\n\n预登记Fisher增量判据：**{geometry_adv}**。需要至少两档尺度同时改善准确率≥0.125、Spearman≥0.15且可分辨覆盖不下降。逐尺度差异见SUMMARY.json。\n'
 text+='\n此处Fisher定义为E_H E_a~p(theta0)[score scoreᵀ]，每个原old_build窗口8个固定模型样本；共1024样本。动作只用于Fisher积分，专家NLL评价标签仍是原专家动作。它覆盖混合权重、均值和尺度，但不是专家标签NLL的Hessian，也不冒充精确GGN。没有用专家梯度外积替代模型Fisher。\n'
 text+='\n两种几何改变了方向身份和等参数范数下的功能变化大小。即使Fisher更好，单个2×4实验也只能支持本定义/采样下的差异，不能唯一归因为“旧功能定义错误”。本轮只有一套固定Fisher模型采样，未验证估计参考依赖。\n'
 text+='\n## 跨轨迹预测，不能与同库校准混算\n\n| 几何 | 尺度 | 符号正确 | Spearman | 新任务同库校准符号正确 |\n|---|---:|---:|---:|---:|\n'
 for kind in primary:
  for scale in p['scales']:
   r=next(r for r in summaries if r['geometry']==kind and r['scale']==scale and r['task']=='old' and r['predictor']=='cross_trajectory')
   new=next(r for r in summaries if r['geometry']==kind and r['scale']==scale and r['task']=='new' and r['predictor']=='same_probe')
   text+=f"| {kind} | {scale} | {r['correct']}/16 | {r['spearman']:.5f} | {new['correct']}/16 |\n"
 text+=f'''\n## 审计与范围

原mean-GN 1倍/1/4倍观测与上一轮逐方向复现见PARENT_PARITY.json。新Fisher算子通过显式小型Jacobian对照，固定样本下的两次特征求解、正交/残差检查见FISHER_NUMERICAL_CHECK.json。数据按原演示划分，不重抽buffer。

主测量{eff['total_seconds']:.2f}秒，Fisher构建{eff['basis_seconds']:.2f}秒，峰值显存{eff['peak_GPU_bytes']/2**30:.3f} GiB。所有模型测量保持解码器eval/float32、头eval/float64，与上轮一致。

描述分类：**{classification}**。本轮不能证明实际AdamW更新、在线控制、持续训练或闭环性能收益；不会自动升级控制器。三档新缩步是本轮事前指定诊断，不回填上一轮主尺度结果。

参考：[Martens (2020), New Insights and Perspectives on the Natural Gradient Method](https://www.jmlr.org/papers/v21/17-678.html)。Fisher与GGN的等价有条件，不能把模型期望曲率直接当作任意专家样本NLL Hessian。

`training_active=false`; `formal_training_resumed=false`。
'''
 (OUT/'FINAL_REPORT.md').write_text(text)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.7))
 for kind,rs in primary.items():
  x=[r['scale'] for r in rs]
  axes[0].plot(x,[r['conservative_sign_accuracy'] for r in rs],'o-',label=kind)
  axes[1].plot(x,[r['spearman'] for r in rs],'o-');axes[2].plot(x,[r['relative_linearization_error'] for r in rs],'o-')
 for ax in axes:ax.set_xscale('log',base=4);ax.set_xlabel('epsilon / epsilon0');ax.grid(alpha=.2)
 axes[0].set_ylabel('Conservative sign accuracy');axes[0].set_ylim(0,1.05);axes[0].legend(fontsize=7);axes[1].set_ylabel('Spearman (same probe)');axes[2].set_ylabel('Relative Taylor error');axes[2].set_yscale('log');fig.tight_layout();fig.savefig(OUT/'SCALE_DIAGNOSTIC.png',dpi=160);plt.close(fig)
 write(OUT/'STATUS.json',dict(stage='COMPLETE',classification=classification,training_active=False,formal_training_resumed=False));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
