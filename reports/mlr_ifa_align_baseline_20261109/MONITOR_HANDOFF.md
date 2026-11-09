# MLR+IFA + ALIGN：部署准备完成，尚未启动

观察：2026-11-09T17:09:18.970265+08:00。文本合同：`6a6298283056986941016d1f487f8169fcbfe760679088e675f51ceaa015f7d7`。

六份代码由各自已接受Random源逐项匹配后适配；保留其模型、IFA、更新顺序、AdamW、逐更新scheduler及原评估器。唯一训练目标变化是ALIGN下一动作与有效窗口BC mask；没有RA/E/R/q/候选门控。Random人口、样本成员、顺序及每stage100轮保持。

实际CPU审计覆盖六case与24stage：特征SHA、原Random成员/顺序、原始HDF5下一动作、完整窗口边界mask、初始状态绑定、阶段人口全部通过。无效窗口留在原人口中，只屏蔽BC，IFA与sampler不删样本。仅复用各自完整current标签，不从RA选样缓存借成员。

代码准备包括：训练及5调用/stage隔离准入、独立端口最多两训练的有限队列、串行原评估器、断言失败停止新启动且不终止已启动训练、进度前缀审计、CPU最终验收及六指标/SF/FS/全部负单元比较器。正常与失败两种队列模拟实际通过，0训练0环境。全部GPU准入和最终CPU包装仍待真实执行，不预先记PASS。

## 启动顺序

当前q1 OBJECT和GOAL全部自然结束 → 两套件独立CPU接受/比较及FINAL_REPORT发布 → 新鲜容量和GPU检查 → 唯一部署 → OBJECT三训练/600正式 → GOAL三训练/600正式 → 本轮独立CPU接受/比较/解释发布。当前控制器未增加新队列或接收信号。

启动入口：`python launch_after_q1.py OBJECT_FINAL_DIR GOAL_FINAL_DIR`。入口核实际LOCAL_FINAL_AUDIT、PRIMARY_S_COMPARISON、已发布FINAL_REPORT与remote MLR SHA，再绑定真实旧合同/退出及无进程；缺任一条件在部署前停止。不要直接重复任何controller或dispatch。实际失败要先保留进程、退出、冻结hash和成本再审计，不自动重训。

## 容量与本地空间

六run历史同形状4CP+1恢复文件合计约3.87 GiB；考虑两训练原子保存临时副本、轻量日志与余量，以6 GiB新输出峰值加2 GiB保留作门槛。16:59实际data可用7.15 GiB，当前不足门槛约0.85 GiB，且q1仍会增长。不能把当前空闲重复当作新实验可用空间，启动时必须重新核验。

只保存/发布源码、标签身份/哈希、选样名单、配置和已有逐回合结果；未下载任何模型、latent、示范动作或恢复状态，未删除任何旧模型。新buffer只链接已核SHA的只读特征payload，避免复制GB级特征。原始只读返回中的大元数据及含动作示例的探查仅本地保留，不发布。

## 当前训练保护

16:59只读核实原q1合同和148冻结项全部一致，controller2638/2639、launcher2641/2642、GPU工作进程2650/2652存在。无暂停、重启、源码替换或旧结果覆盖。新root与private目录均不存在。

## 接受与解释边界

新实验实际更新预算544800、2400轮、0训练期采集、1200正式；隔离准入计划120另记。旧Random1200正式只读复用。当前仅准备接受，不是实验接受；旧硬件和历史global动作随机流不同，最终不将所有差异归为严格单因素因果。保持开发集重复使用与预训练来源未闭合限制；不按结果挑seed/版本或补评。

先读PROTOCOL、PREPARATION_STATUS、SOURCE_REVIEW、INPUT_AUDIT、CONFIGURATION_REVIEW、PREPARATION_VALIDATION、UPLOAD_REVIEW和FINAL_WRAPPER_REVIEW。当前只等条件，不运行任何新GPU诊断。
