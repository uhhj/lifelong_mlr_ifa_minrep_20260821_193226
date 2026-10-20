"""Render only previously collected records/statistics; no experiment execution."""
import json
from pathlib import Path
from fractions import Fraction as F
R=Path(__file__).resolve().parent
a=json.loads((R/"ANALYSIS.json").read_text())
assert a["status"]=="PASS_READONLY" and a["local_input_hashes_after_equal"]
lines=[
"# ZR-v3 Task6下降的三组只读审计",
"",
"结论：KL执行检查确实通过，但它提供的是同输入下、单步所选Gaussian的偏移上限，不是成功率保持条件。现有证据支持：自身残差仍持续改变，频繁投影之后仍可保留不利的修正方向；Task6的最终退步不能简单归咎于单元起始RNG不一致。尚无法定位具体物理动作或独立证明投影导致下降。",
"",
"范围：OBJECT seed41，原RA＋IFA、旧RA＋ZR-v2及本轮RA＋ZR-v3。只读600条既有结果、三组30份RNG快照、两组各10,550条校准batch日志、32次结构检查和解析配置。未加载模型checkpoint、未做模型前向、未初始化CUDA、未创建optimizer或执行环境；仅在CPU解析既有RNG文件。远端41份输入与本地分析输入前后SHA256一致。",
"",
"## 1. Task6下降发生在哪里",
"",
"| 阶段 | RA成功数/20 | v2成功数/20 | v3成功数/20 | v3−RA pp |",
"|---:|---:|---:|---:|---:|"]
for s in range(6,10):
    rows=[r for r in a["triples"] if r["stage"]==s and r["task"]==6]
    n={k:sum(r["success"][k] for r in rows) for k in ("RA","v2","v3")}
    lines.append(f'| {s} | {n["RA"]} | {n["v2"]} | {n["v3"]} | {(n["v3"]-n["RA"])*5:+d} |')
lines += [
"",
"不是各阶段都下降：阶段8有明显改善，阶段9再转差。阶段8→9，RA的Task6为35%→45%，v3却55%→25%。这不能只叫公共父模型遗忘，因为父策略本身在该比较中提高了；但不同阶段父模型、残差及其校准集同时变化，现有记录不能拆开各因素。",
"",
"Task6对总AUC差的贡献为−1.5625pp；总差为−1.770833pp。Task7贡献−0.833333pp、Task8贡献+0.625pp、Task9为0。Task6约占净差的88.2%，仅为带抵消的算术分解，不是因果贡献比例。",
"",
"## 2. 逐episode：是哪些初态发生翻转",
"",
"按同阶段、同Task6、同初态哈希配对。SS/FF为两边同时成功/失败；SF为左成功右失败；FS相反。编号均从0开始。",
"",
"| 阶段 | 比较 | SS | SF | FS | FF | SF初态编号 | FS初态编号 |",
"|---:|---|---:|---:|---:|---:|---|---|"]
for r in a["pairs"]:
    if r["task"]!=6:continue
    c=r["counts"];ids=r["ids"]
    lines.append(f'| {r["stage"]} | {r["left"]}→{r["right"]} | {c["SS"]} | {c["SF"]} | {c["FS"]} | {c["FF"]} | {ids["SF"]} | {ids["FS"]} |')
lines += [
"",
"最终RA→v3：原9条成功中保留4条，失去5条，并新增1条成功，净少4条。RA→v2最终同样失去5条，但新增2条。共同失去的初态是3、12、13、18；v3额外失去9，恢复了v2丢失的10。v2→v3最终只净少1条，因此不能把v3比v2差的5pp夸大为稳定的新失败机制。",
"",
"阶段8，v3保留了RA全部7条成功，并多成功4条；阶段7则失去4条而无新增成功。说明修正没有统一的“只缩小误差”效应，而是随父阶段与状态发生有利或不利变化。这里的SF只表示结果翻转，不直接等价于某次残差动作被证明损坏轨迹。",
"",
"## 3. 三种读出到底改变了什么",
"",
"| 方法 | 新残差 | 校准目标 | 是否投影 | 不变部分 |",
"|---|---|---|---|---|",
"| RA | 无 | 已完成父BC＋IFA | 不适用 | 原父策略 |",
"| v2 | 同一7维预激活偏移施于5个均值 | 全GMM混合NLL | 无 | 同输入下尺度与logit |",
"| v3 | 只修正argmax分量均值 | 所选Gaussian NLL | 实际执行KL≤0.5 nat | 同输入下尺度、logit及其他4个均值 |",
"",
"RA与自身相比的新增残差为0；不能将其和另一个父阶段的差异混成残差。v2没有记录实际执行KL，不能补造v2的KL分布或断言其一定超过v3。v2与v3不仅相差投影，还相差训练目标及分量修正方式；没有运行v3-A，不能从两者的差值提取纯投影效果。",
"",
"阶段结束结构日志（delta_max是投影前、tanh均值前的残差幅值；阶段7–9是所有已见任务混合最大值，不是Task6专属；U范数是参数范数，不等于动作偏移）：",
"",
"| 阶段 | v2 delta最大 | v3 delta最大 | v2 Task6 U范数 | v3 Task6 U范数 | v3全dataset投影数/窗口数 |",
"|---:|---:|---:|---:|---:|---:|"]
for s in range(6,10):
    v2,v3=[next(r for r in a["structure"] if r["method"]==n and r["stage"]==s) for n in ("v2","v3")]
    lines.append(f'| {s} | {v2["delta_max"]:.6f} | {v3["delta_max"]:.6f} | {v2["branch_norms"]["6"]["U"]:.6f} | {v3["branch_norms"]["6"]["U"]:.6f} | {v3["projected_samples"]}/{v3["samples"]} |')
lines += [
"",
"v3的Task6 U范数从0.067225增至0.131273；在阶段7/8/9中，含Task6样本并更新其分支的batch分别为2238/2910/3303。零响应门只隔离其他指令分支，Task6自己的UV仍被旧记忆继续更新，且是在新的父阶段网络上重校准。这里有实证表明它不是冻结的旧任务保护器。",
"",
"全部结构检查中没有非自身门泄漏超阈值、饱和爆炸、非选中均值被改动或父模型误加载证据；同阶段v2/v3公共网络与T0的实际tensor摘要相同。后续阶段没有重新扫描Task6未入库的全数据；不能用混合dataset最大值断言所有Task6状态都有相同幅值。",
"",
"## 4. KL上限守住了，但不是很少干预",
"",
"| Task6阶段 | 投影lane状态比例 | 原始KL最大 | 实际KL最大 | 在原始KL最大状态的投影系数 | v3−RA pp |",
"|---:|---:|---:|---:|---:|---:|"]
for r in a["projection"]:
    if r["task"]!=6:continue
    s=r["stage"];rows=[x for x in a["triples"] if x["stage"]==s and x["task"]==6]
    diff=5*sum(int(x["success"]["v3"])-int(x["success"]["RA"]) for x in rows)
    lines.append(f'| {s} | {r["projected_percent"]:.2f}% | {r["raw_KL_max"]:.4f} | {r["executed_KL_max"]:.12f} | {r["alpha_at_max_raw"]:.4f} | {diff:+d} |')
lines += [
"",
"比例分母含成功后仍参与batch前向的lane，不是活跃失败轨迹的比例，也不是独立episode数。阶段9最大原始KL约73.08，对应系数约0.0827，确实有很强收缩；这些是未执行候选的统计，不能把73.08说成实际违规KL。",
"",
"关键数学含义：同尺度Gaussian下，KL=½Σ_j[(μ'_j−μ_j)/σ_j]²。因此0.5上限允许标准化均值偏移的L2范数达到1，并不是“动作只改0.5%”。投影触发时通常把候选压到边界，而非回退至父动作；它保留候选方向，不能判断该方向是否有利于抓取或轨迹稳定。这里没有统一的最大实际动作偏移值，因为它还取决于各维σ与tanh，日志没有逐维尺度。",
"",
"每一步都重新使用0.5预算，并没有整条轨迹的累计预算。即使每步偏移不大，也可能经环境反馈形成不同观测与后续父输出。KL比较的是v3当前输入处的父输出与v3输出，不是RA独立rollout同时刻的不同观测输出；因此不能把“同输入分布接近”解释为“两条轨迹一直接近”。",
"",
"阶段8、9投影比例与成功率变化方向不一致，且阶段9Task7投影约86.14%仍达到95%成功。现有聚合数据不支持“投影越多，表现越差”的单调解释。投影率和最大KL没有按episode保存，不能把它们分配给SF/FS样本作相关性分析。",
"",
"投影外侧的解析均值近似为base+c·shift/||shift/σ||；同方向继续放大shift不会继续增加执行偏移。这使原始残差参数增长与实际执行幅度脱钩。它是源码可推导的性质，不代表已证明优化器因为该性质失败。",
"",
"## 5. 所选分量目标更接近执行规则，仍没有直接优化成功率",
"",
"Task6阶段只包含Task6窗口。v3 epoch0→9的训练NLL从1,274,021.16降至1,274,018.07，而当期成功率60%→55%；v2对应混合NLL为−31.15→−32.31，成功率60%→45%。两种NLL定义不同，绝不能比较绝对数值判定谁更好。epoch0已经训练过一轮，也不是零残差损失。",
"",
"大幅加权误差说明这个冻结的所选Gaussian拟合专家标签仍很吃力，但没有逐维σ、分量责任度或逐样本损失，不能判定是否由极小σ、少量离群样本或特定动作维度主导。Gaussian目标按1/σ²加权误差，本身不包含任务成功、接触安全或父成功轨迹保持；投影限制可动范围，也没有改变这个目标。阶段7–9日志是新旧混合loss，不能宣称Task6专属loss持续下降。",
"",
"## 6. 对Task6，随机流错位不是已发现的原因",
"",
"三份resolved_actual评估配置逐字段比较，唯一差异为输出路径。200组三方法结果的初态哈希、modal与同阶段父checkpoint绑定一致。30份原有RNG文件只在CPU解析：前三组在前8个cell的Python/NumPy/CPU/CUDA状态全部一致，仅最后阶段Task8/9有CUDA差异。",
"",
"尤其四个Task6 cell都位于阶段开头：三组完整起始RNG一致，两个vector均600步，10lane固定批形状。因此没有发现Task6因之前任务提前终止造成的CUDA流偏移，第二批也没有第一批长度不同带来的偏移。结合共用采样调用路径，这比一般的“只配同seed”更有可比性。",
"",
"但未记录逐步标准化采样噪声、观测及环境状态，仍不能宣称逐步动作/环境轨迹完全配对；相同起始随机状态不消除有限20次评估波动。可以排除已观察到的单元起始RNG不一致，不能把单次结果升级成稳定的因果估计。",
"",
"## 7. 可支持的判断与不能支持的判断",
"",
"更有证据支持：保护了跨指令隔离与单步分布偏移，不等于保护自身成功行为；Task6自身残差在不断重校准，投影常被激活，但允许的修正仍有方向风险与闭环反馈风险。阶段8收益、阶段9损失并存，说明并非模块始终无效，而是没有稳定的收益保持。",
"",
"目前不能确定：具体是夹爪、位置还是旋转维度导致失败；失效发生在哪个控制时刻；是投影太强或太弱；换epsilon即可修好；是否过拟合；新增Task9分支是否在未记录闭环状态泄漏。已有训练窗口结构不支持泄漏解释，但没有逐步门响应日志就不能将其扩展为全轨迹证明。",
"",
"本轮不改变epsilon、任务预算、训练目标或分支，不启动v3-A或新seed。全部结论仅来自既有单seed探索性数据，v3与v2组合改动和未隔离轨迹随机流仍是解释边界。",
"",
"## 8. 全20初态的Task6结果历史",
"",
"四位字符串依次对应阶段6/7/8/9，S成功、F失败；跨阶段父模型不同，不能把一行当同一模型的四次重复。",
"",
"| 初态编号 | RA | v2 | v3 |",
"|---:|---|---|---|"]
for r in a["task6_history"]:
    lines.append(f'| {r["trial"]} | {r["history"]["RA"]} | {r["history"]["v2"]} | {r["history"]["v3"]} |')
lines += [
"",
"## 9. 复算与证据",
"",
"```powershell",
"python execution/20261020_zrv3_object_seed41/task6_readonly/fetch_readonly.py",
"python execution/20261020_zrv3_object_seed41/task6_readonly/analyze.py",
"python execution/20261020_zrv3_object_seed41/task6_readonly/write_report.py",
"```",
"",
"退出码均0；远端只执行`/data/mlr_ifa/env/bin/python -`读取既有文件，脚本通过stdin传入，未在实验目录写文件。REMOTE_COMMAND.json保存命令/脚本哈希；REMOTE_AUDIT.json保存41文件前后哈希证明、30份RNG摘要和batch聚合，不含权重或RNG二进制。ANALYSIS.json包含全部200组三方法逐episode、30行成对汇总、10cell投影、8阶段结构比较、全部任务指标/精确AUC分解、源码与本地输入哈希。TASK6_EPISODE_HISTORY.json提供20行完整初态历史。",
"",
"源码依据：template/attachment/execution_core.py、template/source/zr3_exp/{model,calibrate,diagnostics,evaluate}.py；历史v2的server_auto/zr_template/source/zr2_exp/model.py。当前报告不覆盖旧报告或实验验收。",
]
(R/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print("REPORT.md generated from existing audited evidence")
