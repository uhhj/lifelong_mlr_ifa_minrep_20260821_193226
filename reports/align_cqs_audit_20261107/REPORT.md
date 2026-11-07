# 历史 RA-Coreset+ALIGN 与最新 C/Q/S：实现与协议核对

结论：历史ALIGN不是最新C。原拟四行“RA → RA+ALIGN → 加成功历史保持(Q) → 加门控(S)”遗漏了C/Q/S共同增加的执行专家代价与margin路由辅助项，不能作为只切换保持和门控的严格勾选表。已有数据足以补成五行累计配置表；最直接的完整机制对照仍是最新C/Q/S。无需为此次核对重训或重评。

## 1. 由实际源码确定的损失关系

令 L_A = L_BC(aligned a[u+1], valid mask) + 0.1 L_IFA。

- 历史ALIGN：L_A。observe.py中ALIGN分支直接令loss=native，没有执行专家监督或新增保持项。
- C：L_A + 0.1 E，其中E为当前batch中“旧任务且BC合法”窗口的 mean(c_exec + 0.1 margin_route)。c_exec包括所选GMM分量的归一化动作均值平方误差与方差；路由以最小代价分量为目标，margin=0.1。三组动作尺度按合法旧训练动作RMS、floor=0.1确定。C虽保留历史教师准备代码，但C分支的损失不使用教师值。
- Q：L_A + 0.1 E + 0.1 (R_q + 0.1 H_q)。R_q/H_q使用自己的成功轨迹缓存及任务获得时教师，q=独立成功采集轨迹数/10，4槽分母不重归一化。关闭生产候选门控，始终执行Q更新。
- S/Ours：共同完整状态起点生成C和Q候选；每10步及轮尾短块比较。Q的当前任务BC与执行专家探针不劣于C、旧任务支持加权保持探针严格更好时接受Q，否则接受C。不是对Q再加一个常数损失项。

上述E与R都有自己的路由项，不能混为同一个路由模块。Task6没有旧窗口或旧成功缓存，E/R都不生效；因此Task6模型相等不证明Task7–9目标相同。

证据位置：历史additions/caet_v3_exp/observe.py和model.py；C的updates.py/losses.py/legacy_losses.py；Q v2的updates.py/support_memory.py；S的fork_tools.py。AUDIT.json保存源SHA和实际行号摘录。

## 2. 本次实际复核覆盖

本地读取已完成接受证据，未连接服务器，未加载模型/状态/缓存载荷，0新训练、0前向、0环境。

- 六case：OBJECT与GOAL各seed41/42/43；历史ALIGN均已独立接受，C/Q/S最终接受文件与EVIDENCE源SHA对应。
- 24阶段：历史ALIGN、Q、S的完整训练manifest SHA相同；C的五个扩展run共20阶段以PARENT_BINDING及历史冻结合同SHA对应。GOAL41 C通过相同历史O路径与manifest加载源码对应，其最终轻量包没有另存逐阶段manifest SHA，不把这四阶段称为额外完成了独立hash重证。
- 历史manifest包含有序训练成员、样本SHA、任务身份和BC有效mask。Q/S与C均加载原ALIGN的下一步动作标签；没有重选RA成员、丢弃无效窗口或更换BC标签。
- 24阶段每阶段100完整epoch；实际日志的epoch/batches/lr_first/lr_after逐项相等。当前C/Q/S逐epoch sampler SHA相等。没有声称已逐步比较全部模型轨迹。
- 18组C/Q/S官方初始模型tensor hash均与对应历史ALIGN一致。历史CUDA RNG相等的已接受记录也全部为True。六个case的C、Q、S Task6最终模型对历史ALIGN全tensor相等记录均为True。
- 1,200个stage/task/initial_index逐条核历史ALIGN与C/Q/S的初态文件及实际初态SHA相同；当前C/Q/S的noise SHA、env_seed、attempt逐条配对相同。
- 六case中Q与S的updates.py、losses.py、support_memory.py、memory.py、fork_tools.py、legacy_losses.py按UTF8 LF文本逐文件相等。Q的生产循环直接update(...,'Q')，不执行gate，且断言没有GATES记录。

## 3. 相同设置与实质差异

|项目|历史ALIGN与最新C/Q/S的核对结论|
|---|---|
|结构/主输入|同一历史ALIGN来源：8帧特征、Temporal Transformer、5分量7D GMM、FP32、batch10|
|初始化|各suite官方初始化，模型tensor相同；GOAL官方ep96、OBJECT官方ep99|
|数据/主BC|同seed原RA成员与顺序、a[u+1]、合法mask、无效BC窗口仍参与人口和更新预算|
|IFA|保留0.1系数；实际OBJECT gamma=0.3，GOAL gamma=0.7，从各自历史配置继承|
|优化器|AdamW lr1e-4、wd0.1、warmup500、clip100；每stage100轮，实际LR日志相同|
|基础辅助损失|历史ALIGN没有E；最新C/Q/S均有E，这是额外机制|
|环境采集|ALIGN/C为0；Q/S每stage10次，共40次/run，自己的成功缓存|
|接受更新|GOAL76100、OBJECT105500/run，普通C/Q/S一致|
|物理optimizer调用|C/Q为76100/105500；S为142200/195300（GOAL/OBJECT）|
|正式评估规模|相同四阶段三角10格，每格20初态，200回合/run，600步上限、5zero|
|动作分布|保持argmax GMM分量的Normal执行语义|
|动作随机流|历史ALIGN按stage重置global RNG；新C/Q/S用按seed/task/initial_index生成的固定逐轨迹噪声；历史与新结果不具备逐轨迹噪声相同证据|
|推理实现|新评估显式缓存单帧前端特征，长度1–8独立准入按1e-4容差核对；不是历史evaluate.py逐字节复用|
|硬件|存在历史执行硬件差异；初始化及Task6相等不能替代闭环环境/推理完全受控证明|

C扩展PROTOCOL中有一处泛称“IFA gamma0.7不变”，但实际runtime读取各自历史resolved配置；本次六case实际配置证据为OBJECT0.3、GOAL0.7，未观察到将OBJECT改成0.7。此处据实际来源核对，不以泛化文字覆盖配置。

等优化器预算C是另一组控制，Task6=100、后续各200轮，不能混作上述普通C行；可在补充计算预算分析中单列。

## 4. 推荐表格结构

所有行均以复现MLR+IFA为基础，可不额外增加基础行，但在表注写明。使用明确的机制列：

|配置|RA-Coreset|ALIGN|执行专家监督E|成功历史q加权保持R|门控G|
|---|---|---|---|---|---|
|RA-Coreset|✓|—|—|—|—|
|RA-Coreset+ALIGN|✓|✓|—|—|—|
|+执行专家监督（C）|✓|✓|✓|—|—|
|+成功历史保持（Q）|✓|✓|✓|✓|—|
|完整方法（Ours）|✓|✓|✓|✓|✓|

可以放在一张表，但应称“组件配置与机制对照”。历史前两行和最新后三行之间用分隔线/简短表注说明评估随机流版本；它们不构成严格逐轨迹配对的单开关因果链。若正文要求最简洁、最受控的机制消融，正文只放C/Q/Ours，RA/ALIGN另设一个小panel，二者共享同一指标列。

若论文将E归入SBHR/SHBR整体，则C已是该整体的部分机制，Q是w/o gate；不能将C或Q误标为只有历史ALIGN，或将Q−ALIGN的全部差值归于R。论文方法章节需要明确E归属，再统一命名。

## 5. 已有指标的交叉核对参考

以下为接受结果中的三seedAUC均值，展示这些行确为不同已完成配置，不重新选结果：

|配置|OBJECT AUC|GOAL AUC|
|---|---:|---:|
|RA-Coreset|73.2292|81.9792|
|RA-Coreset+ALIGN|80.1736|81.1111|
|C（普通预算）|85.0000|82.5000|
|Q|84.6181|83.5069|
|Ours|86.4931|87.2569|

所有完整FWT/NBT/FinalAll和逐seed值仍按20261101_ablation_summary/DATA.json取数；表格编制时重新复算标准差。没有根据本次表格目的更换历史seed43或任何实验版本，也没有新增科学运行。
