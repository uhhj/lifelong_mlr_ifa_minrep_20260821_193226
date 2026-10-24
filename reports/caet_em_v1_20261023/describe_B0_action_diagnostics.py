"""Render audited existing-checkpoint diagnostics without additional model execution."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;O=Path(sys.argv[1]);O=O if O.is_absolute() else R/O
def js(p):return json.loads(p.read_text(encoding='utf8'))
s=js(O/'SUMMARY.json');p=js(O/'PAIRED_HORIZONS.json');a=js(O/'LOCAL_AUDIT.json');e=js(O/'EVIDENCE.json')
assert a['status']=='PASS_LOCAL_B0_ACTION_SCALAR_AUDIT_NOT_CLOSED_LOOP_ACCEPTANCE'
lines=['# B0已有checkpoint：分步动作、GMM分量选择与方差诊断','','结论：在已见训练/旧RA分布上，未发现第一步普遍比未来三步更差；最显著的误差来自少数夹爪大幅均值偏差及分量选择。混合分布可解释目标，不保证最高权重分量正确。方差对平移/旋转采样误差有贡献，但几乎不解释大幅夹爪偏差。这些发现尚不能解释全部闭环差异。','','本轮使用四个B0已有完整checkpoint，每个104键实际CPU严格加载。独立单线程CPU低优先级FP32/eval/no_grad运行，GPU未启用、0优化器更新、0环境执行、0服务器产物写入，主实验科学源和模型参数不变。已有冻结输入与四个CP前后SHA绑定；不改变主队列或监控安排。CPU/I/O消耗仍单独计入，不能称零资源影响。',f"共{a['cost']['windows']}个窗口、{a['cost']['forward_calls']}次batch前向、{a['cost']['valid_positions']}个合法未来动作位置；CPU {a['cost']['cpu_seconds']:.3f}s，远端诊断墙钟{a['cost']['wall_seconds']:.3f}s。40个task/stage/horizon标量文件本地独立验算通过。原GPU四步诊断任务均值与本次CPU复算最大差{max(x['absolute_difference'] for x in a['CPU_GPU_parity']):.3g}，小于预登记容差。全过程没有读取/下载模型payload到本地。",'', '## 第一步并未普遍劣于未来三步','','下表只取各任务刚学完时的当前任务，且限制为同一样本四步均合法，排除不同尾部mask人口影响。指标为原动作单位的分组Huber，不是毫米位置误差。窗口重叠且来自训练演示，不作独立样本显著性检验。','','|任务|相同完整四步窗口|第一步误差|未来2–4步平均误差|','|---|---:|---:|---:|']
for t in range(6,10):
 x=p[f'stage{t}_task{t}'];lines.append(f"|Task{t}|{x['common_four_valid_windows']}|{x['first_mean_huber']:.8f}|{x['future3_mean_huber']:.8f}|")
lines+=['','四个当前任务第一步均值都更小，因此先前“第一步被四步目标普遍牺牲”的猜测没有得到该离线证据支持。不能反向断言四步目标有益：还没有同初始化单步对照，也没有独立演示或闭环访问状态上的分步误差。旧任务存在例外，例如最终Task7第一步0.00216668、未来三步0.00075946；全部分步结果保留在SUMMARY与PAIRED_HORIZONS。','','## 分量选择与少量夹爪大误差','','下表使用各任务刚学完时全部合法第一步位置。事后oracle是在已有五个分量里按专家目标选择最小均值误差，只用于分析，不可部署、不是新闭环结果。','', '|任务|第一步合法数|默认分量Huber|事后最佳分量Huber|非最小误差分量比例|夹爪绝对误差>1计数|夹爪对总Huber贡献|','|---|---:|---:|---:|---:|---:|---:|']
for t in range(6,10):
 x=s[f'stage{t}_task{t}_first'];v=x['statistics'];c=x['error_concentration']
 lines.append(f"|Task{t}|{x['count']}|{v['huber']['mean']:.8f}|{v['oracle_huber']['mean']:.8f}|{100*x['selected_not_oracle_mean_fraction']:.2f}%|{c['gripper_abs_error_gt1_count']}|{100*c['gripper_huber_share']:.2f}%|")
lines+=['','Task7第一步8/1400、Task9第一步5/1239存在夹爪绝对误差>1；夹爪维度分别贡献94.14%和92.71%的分组Huber。这里不是把所有非oracle分量都定义成严重错误：非oracle比例12.0%和5.81%还包含很小的连续动作差异。Task7大夹爪误差样本的最高权重均值0.4396、前二权重差0.1373；Task9对应0.6085、0.2703。说明部分严重偏差伴随分量歧义，但不能用单一置信阈值自动修复。','','GMM训练目标是mixture NLL，部署选最高logit分量再采样。Task7第一步mixture NLL均值−30.1838，但只用所选分量解释专家目标时NLL均值约1.143×10^6；Task9对应−31.8603与8.071×10^5。巨大差距由少数目标与所选窄夹爪Normal严重不符的样本支配，不是训练损失爆炸或数值溢出。连续密度NLL可以为负。另一分量解释目标，使mixture训练拟合良好，并不保证argmax分量的行为正确。此训练/部署语义原RA也存在，不能说B0新引入了该语义，或仅凭此证明B0相对RA下降的原因。','','## 方差：连续动作有贡献，夹爪大误差主要不是采样噪声','','不产生随机采样。对固定目标和选中Normal，独立解析复算E[(a−target)^2]=(mu−target)^2+sigma²。下表均为第一步；平移sigma RMS汇总各位置/坐标，覆盖率按每个坐标统计。','','|任务|平移sigma RMS|平移±2sigma覆盖|方差占平移采样期望MSE|夹爪sigma中位数|夹爪±2sigma覆盖|','|---|---:|---:|---:|---:|---:|']
for t in range(6,10):
 x=s[f'stage{t}_task{t}_first'];v=x['statistics'];z=x['translation_variance_analysis']
 lines.append(f"|Task{t}|{z['sigma_rms']:.6f}|{100*v['translation_coverage2']['mean']:.2f}%|{100*z['variance_share_of_expected_sample_mse']:.2f}%|{v['gripper_sigma_rms']['median']:.8f}|{100*v['gripper_coverage2']['mean']:.2f}%|")
lines+=['','Task7/9夹爪预测方差约1e−8，而均值MSE分别约0.022857与0.016142，采样方差贡献不到百万分之一。已有训练分布上的夹爪大偏差主要由所选分量的均值造成；仅减少随机噪声不会纠正均值选错。夹爪近二值命令会自然得到很小的条件方差，因此贴近min_std并不单独证明病理性方差塌缩。','','平移±2sigma覆盖约84%–92%，低于单高斯95.45%的参照，但最高权重分量不一定解释当前专家动作；覆盖不足同时混合了分量选择、均值偏差和尺度校准问题。不能将它全部归因于方差太小，也不能因此改变sigma。各组MAE/MSE、sigma分位数、1/2sigma覆盖、分量熵/权重差及NLL完整保留。','','解析[-1,1]越界概率约7.14%的全坐标平均主要可由一个夹爪维度处在边界附近、Normal约一半概率落在边界外解释，不是“7.14%动作会失败”或大方差证据。未更改裁剪、采样或部署行为。','','## 仍不能解释的闭环差异','','Task6相同旧RA样本的第一步Huber从stage8的0.00145270进一步降至stage9的0.00006251，夹爪大误差由2/764降到0/764，平移采样期望MSE亦从0.00101648降至0.00051850；其闭环却95%→65%。因此不只是四步平均掩盖了第一步退化：已有旧RA上的第一步均值、方差相关误差也改善了。未见状态上的行为、分布偏移、关键时刻的错误累积仍未直接观测，不能从这些离线标量作唯一因果归因。Task8也有类似少量夹爪偏差但闭环95%，所以它们不能直接换算为任务失败率。','','本次数据不是独立测试演示。所有oracle选择依赖专家目标，不能作为改部署规则的依据；样本重叠、单seed、重复探索限制保留。B0完整闭环记录仍为原200条，无新增rollout；整组三臂最终验收状态不因本诊断改变。没有自动开展重训、消融或修复策略。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
binding={name:hashlib.sha256((O/name).read_bytes()).hexdigest() for name in ('EVIDENCE.json','LOCAL_AUDIT.json','SUMMARY.json','PAIRED_HORIZONS.json','REPORT.md')}
binding['diagnostic_protocol_sha256']=hashlib.sha256((R/'B0_ACTION_DIAGNOSTIC_PROTOCOL.md').read_bytes()).hexdigest()
(O/'REPORT_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
print(O/'REPORT.md')
