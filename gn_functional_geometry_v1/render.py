from .common import *
def main():
 h1=read(OUT/'H1_REPORT.json');h2=read(OUT/'H2_REPORT.json');h3=read(OUT/'H3_REPORT.json');s=read(OUT/'SUMMARY.json');eff=read(OUT/'EFFICIENCY.json');n=read(OUT/'GN_NUMERICAL_CHECK.json');p=read(OUT/'PROTOCOL.json');a=h1['by_scale'][-1];g=h3['primary']['GN'];r=h3['primary']['RANDOM']
 text=f'''# GN 功能几何最小机制实验 v1

**结论：{s['result']}；总体推进门未通过。** H1、H2获得本轮局部工程证据，H3未达到预登记要求。没有修改正式模型、buffer或solver，没有执行优化器更新、完整训练或闭环。

## 实验范围与冻结规则

- 原生MLR seed45，task8 ep99，尚未进入task9；不是失败后的内存状态。
- 只分析策略头2,701,387个参数。功能输出是GMM混合期望动作，不是最大权重分量，也不是随机动作。
- 保留官方terminal-frame完整GMM NLL。解码器eval/float32；所有诊断分支的头统一使用float64。原生接口对照见MODEL_INTERFACE_CHECK.json。
- 旧任务：GN构建128窗口/4演示，关系梯度64窗口/2演示，评价128窗口/4演示。三组演示不重叠。新任务：梯度和评价各64窗口/5演示，演示不重叠。
- 旧窗口均属于进入task9时冻结plan的实际保留成员。原始HDF只用于核对演示及窗口来源，没有补入已丢弃旧样本。
- 动作权重从old_build专家动作确定：平移、旋转、夹爪分组等权，组标准差下限0.05，未使用评价数据调尺度。
- 主扰动范数为{p['epsilon']:.9g}；0.25/0.5倍仅作预登记稳定性检查。这是六个历史原生AdamW状态推算的舍入前头位移中位数，来自同seed另一原生载体，**不是当前checkpoint已保存的精确单步位移**。
- H3各方向使用eta_k=epsilon/abs(z_new[k])实现等参数范数比较，保留下降符号；弱方向不放大。这检验局部方向处理，不等于把实际AdamW位移原大小逐分量投影。
- 门槛在PROTOCOL.json中、在本轮结果测量前冻结；未因结果调整rank、epsilon或门槛。新增NLL单元测试和独立汇总复核未引入新科学候选。

## 1. GN高能量方向是否比随机方向更敏感？

**{h1['status']}。** 独立旧任务轨迹上的GN预测与动作变化，Spearman={a['spearman']:.4f}，Pearson={a['pearson']:.4f}。

| 等范数方向 | 主尺度动作变化中位数 |
|---|---:|
| GN top8 | {a['medians']['GN_TOP']:.8g} |
| 与top8正交的随机方向 | {a['medians']['RANDOM_ORTH_TOP8']:.8g} |
| top32正交补随机方向 | {a['medians']['COMPLEMENT_TOP32']:.8g} |

Top/random约{a['top_random_ratio']:.3g}倍。这个巨大比值也反映270万维空间中随机向量很难对准敏感方向，不能解释为相同比例的方法收益。正交补并没有被证明是真正null空间。三档尺度的排序基本稳定。

## 2. 旧窗口是否集中在少量功能方向？

**{h2['status']}。** 分母为逐输出梯度累计的完整trace={h2['full_trace']:.8g}，不是top32之和。

| Rank | 完整trace能量占比 |
|---|---:|
'''
 for rank,value in h2['captured_energy'].items():text+=f'| {rank} | {100*value:.3f}% |\n'
 text+=f'''
128个构建窗口中，r80={h2['r80']}、r90={h2['r90']}；r95={h2['r95']}。这些是加权期望动作Jacobian的谱能量，不能推广为完整GMM分布或整个模型的功能秩。

## 3. 相邻窗口是否更冗余？

| 关系 | 配对数 | 指纹cosine中位数 | 均值 |
|---|---:|---:|---:|
'''
 for kind,z in h2['pair_distributions'].items():text+=f"| {kind} | {z['n']} | {z['median']:.6f} | {z['mean']:.6f} |\n"
 text+='\n四条构建演示的相邻减跨轨迹均值差均为正。相邻窗口共享7/8输入帧；此结果支持功能重复结构，不能单独证明低秩是由滑窗导致。配对重复使用窗口，不能把数千对当作数千个独立样本。能量指纹丢失符号和动作维度内的朝向。\n'
 text+=f'''
## 4. 符号能否预测真实旧任务改善/恶化？

**{h3['status']}。** 下表为预登记1.0倍主尺度、独立轨迹评价。

| 指标 | GN | 随机等秩 |
|---|---:|---:|
| 非弱方向数 | {g['nonweak']}/16 | {r['nonweak']}/16 |
| 旧任务效应符号正确率 | {g['old_sign_correct']}/16 ({100*g['old_sign_accuracy']:.2f}%) | {r['old_sign_correct']}/16 ({100*r['old_sign_accuracy']:.2f}%) |
| 预测/实测Spearman | {g['spearman']:.4f} | {r['spearman']:.4f} |
| 预测/实测Pearson | {g['pearson']:.4f} | {r['pearson']:.4f} |
| 同评价样本梯度的Spearman（仅Taylor校准） | {g['matched_probe_spearman']:.4f} | {r['matched_probe_spearman']:.4f} |
'''
 for key in ['SHARED','CONFLICT']:
  x,y=g['classes'][key],r['classes'][key]
  text+=f"| {key}满足新旧联合行为 | {x['joint_supported_count']}/{x['n']} | {y['joint_supported_count']}/{y['n']} |\n"
 text+='\n“共享”要求新旧损失均不恶化；“冲突”要求旧损失增加且新损失下降。GN所判共享方向并没有稳定产生共享效果。弱效应不算安全，中性实测不从准确率分母移除。\n\n'
 text+='| 扰动尺度 | GN符号正确率 | GN Spearman | GN同库Taylor Spearman | 随机符号正确率 |\n|---|---:|---:|---:|---:|\n'
 for row in h3['by_scale']:
  x,y=row['GN'],row['RANDOM'];text+=f"| {row['scale']} | {x['old_sign_accuracy']:.4f} | {x['spearman']:.4f} | {x['matched_probe_spearman']:.4f} | {y['old_sign_accuracy']:.4f} |\n"
 text+='\n三档均未建立GN符号判别优势。即使使用评价库自身梯度，一阶预测与有限扰动NLL也不吻合，因此不能把落差全部解释成跨轨迹泛化问题。这与敏感方向上局部线性近似范围较窄相容，但本轮没有单独识别其因果来源，不能断言唯一原因。没有在看到结果后缩小epsilon补救。\n'
 raw=h3['raw_baseline']['results'][-1]
 text+=f'''
## 5. GN是否优于随机子空间及整体梯度基线？

没有建立该结论。GN主尺度符号正确率低于随机，Spearman也更低。由于等参数范数并不等于等功能变化，不能进一步概括为随机空间普遍更好。

整体梯度cosine={raw['cosine']:.6f}；主尺度预测旧损失变化={raw['pred_old_change']:.6g}，实测旧变化={raw['measured_old_change']:.6g}，新变化={raw['measured_new_change']:.6g}。整体基线在此处也没有正确预测保持效果。但只有一个checkpoint/一个cosine，不能对它进行跨状态预测能力的统计比较；三个步幅不是三个独立实验。

## 数值、资源与审计

- basis正交最大误差：{n['orthogonal_max_error']:.3g}；最大相对特征残差：{max(n['eigen_relative_residuals']):.3g}。两次独立初始化子空间一致。
- 小型显式Jacobian对照、原GMM NLL/期望输出接口、方向导数单元测试全部通过。原始CSV独立重算、分割和定向源文件/输入hash复核见INDEPENDENT_AUDIT.json。
- 首次basis计算{eff['basis_seconds']:.2f}秒；重复及残差核验{eff['basis_check_seconds']:.2f}秒；完整trace {eff['trace_seconds']:.2f}秒。
- 主GPU测量约{eff['total_seconds']:.2f}秒（不含CPU资产审计、开发、后处理）；峰值显存{eff['GPU_peak_bytes']/2**30:.3f} GiB；basis {eff['basis_bytes']/2**20:.2f} MiB。
- 初次测量结束时报告模块尚未落盘，汇总导入报错；全部测量产物已经保存。之后只补做报告汇总与CPU独立审计，未重跑模型测量。原run.log保留该过程。

## 最终范围与决策

H1/H2支持：当前头部的加权期望动作敏感结构可估计且集中。H3未通过：这些方向上的新旧梯度符号，在本轮冻结的有限步长和跨轨迹评价下不足以可靠判断共享/冲突。

不据此宣布梯度方向导数理论错误，也不能宣布GN本身无用。GMM方差分支对期望动作的Jacobian为零，但专家NLL依赖它；本轮几何本来就不是完整目标曲率。

仅一个开发seed、一个任务对，不是独立确认集；没有真实AdamW步投影、持续更新、float32写回或闭环收益证据。**当前不推进到MLR/IGFA控制器集成。**

`training_active=false`；`formal_solver_modified=false`；`formal_training_resumed=false`。
'''
 (OUT/'FINAL_REPORT.md').write_text(text)
if __name__=='__main__':main()
