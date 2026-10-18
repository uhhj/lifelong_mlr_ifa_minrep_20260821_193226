# SC-Coreset GOAL seed42 实施与启动报告

状态PARTIAL，正式训练已启动，不能据预检构成变化宣称性能提升。

观测：2026-10-18T17:10:46.436093+08:00，Task6已完成15/100 epoch、1500步；总预算400epoch/76100step/4ep99CP/200正式闭环20vectorattempt。RTX3090，pipeline PID39074、训练PID39083，正常运行。

用户确认先seed42；AUC严格高于同seed原Coreset(78.75%=63/80)，最终旧任务均值不低于原Coreset(68.333333%=41/60)，且全部工程验收通过，才补43/44。Random AUC80.9375%作为参考，不纳入推进门槛。已有两种基线不重训、不重评。

实现限定在当前任务入库，旧buffer不重选；模型、原BC+IFA、gamma.7、任务配额、完整遍历和优化/评估协议保持父实现。初始实际tensor/RNG摘要已与父匹配（正式训练入口断言通过）；Task6 ep99入库前完整tensor比对将在阶段结束执行，尚不提前宣称通过。

四任务监督身份通过完整8×7动作窗口匹配，不把文件号当时间；Task9仅两条padding起点歧义，所需demo/target/length/末端action仍唯一，保留为独立训练样本。分组是时间四分位+真实终点×末端夹爪符号，按完整窗口联合表floor/ceil，取整依次最小交换数、最小比例误差、固定group ID。保留原Coreset，欠额补最远点、超额删最小删除后全局半径点；新点替换被删点原选择槽位，其他相对顺序不动。0交换则原有序列表完全不变。实际batch内容可能改变，IFA按新buffer计算并记录变化。

|任务|N|配额|最少交换M|原覆盖半径|SC覆盖半径|
|---|---:|---:|---:|---:|---:|
|6|997|499|70|0.278769|0.337656|
|7|889|445|38|0.261264|0.293265|
|8|904|452|38|0.290898|0.322380|
|9|1969|985|39|0.241957|0.262762|

全部联合组满足每组floor/ceil及误差<1/q，不仅是两个边际。四个任务所有10条示范在SC中均有成员；完整示范分布见LOCAL_LAUNCH_ACCEPTANCE及selection_preflight。以下边际仅便于读者理解。

|Task6属性|完整窗口|原Coreset seed42|SC|
|---|---:|---:|---:|
|末端正夹爪命令|41.4243%|54.7094%|41.4830%|
|真实终点目标|8.0241%|2.8056%|8.0160%|

用户引用54.84%为既有跨seed审计概述，本轮真实seed42为54.7094%，未将旧均值当输入。SC保留429/499个原Task6成员；覆盖半径增加约21.1%，是监督构成约束与几何覆盖的真实取舍，不能声称几何覆盖也改善。

验证：本地/目标环境5项性质测试通过（含穷举配额最优、30随机问题暴力交换对照、无变化路径、平局、零命令与罕见组）；四任务原S0有序ID与父存储精确匹配、全真实payload配额和最小交换及删除半径通过。目标PyTorch1.13.0+cu116/RTX3090执行恰好4次隔离原生GPU optimizer调用，模型更新与scheduler/RNG恢复一致，通过；0环境步骤、0预检策略rollout。未重新执行父已通过的空间模态/环境检查；模型与评估核心等7个组件字节一致有独立证明。正式另起进程重置seed42，未接续smoke。

真实命令与退出码：

```sh
source /root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/env.sh
/data/mlr_ifa/env/bin/python -m unittest -v sc_exp.test_selection  # exit 0
/data/mlr_ifa/env/bin/python -m sc_exp.prepare                    # exit 0
/data/mlr_ifa/env/bin/python -m sc_exp.freeze                     # exit 0
/data/mlr_ifa/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 -m sc_exp.preflight  # exit 0
nohup /data/mlr_ifa/env/bin/python -u /root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/pipeline.py > "$CORESET_OUT/pipeline.log" 2>&1 < /dev/null &
```

启动shell exit0仅表示后台启动成功，不是实验完成；pipeline/train/evaluate/audit退出码目前待定。流水线将依次完整训练、原协议独立进程评估、只读审计。正式产物：`/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018`；隔离源：`/root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/source`。父数据与缓存保留原位置只读，沿用GOAL新产物系统盘授权。

按首阶段约2.7秒/100步及同GPU既有GOAL训练约40分钟、评估约67分钟估计，预计本次训练和评估约1小时45分至2小时15分，约19:00–19:25 CST结束（仅早期估计，后续更新）。已创建单一两小时监控sc-coreset-goal-seed42；可恢复工程错误按hand-off保留证据恢复，不能盲重启部分全局RNG评估或改冻结科学合同。

新feature/modal来源字节保持父证据不变，SC单独训练合同追踪变化。原预训练state字段缺失仍unknown；调度/ep99为已披露补全。单seed探索性、原global动作RNG未逐轨迹隔离、IFA间接变化等边界保留。离散联合表保持不保证连续动作/状态分布或闭环成功率。本轮没有额外权重/损失评分/样本复制/回放配额改变。

发布内容仅代码、报告、轻量哈希、真实样本ID/时间/group和选择诊断。binding_metadata明确删去原末端动作，未下载发布原bindings、权重、演示、latent、buffer载荷或凭据。原历史报告和用户修改保留。
