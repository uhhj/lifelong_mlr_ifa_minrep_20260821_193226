

## 工程02接续

原完整v1在B Task6完成100轮/10000步并实际tensor精确复现ALIGN后，于Task7教师准备前因CPU/GPU张量直接torch.equal报错退出。check_readonly_20261025_000623已绑定退出1、无进程/GPU空闲及44hash不变。0 Task7更新、0正式环境。新独立engineering02仅将比较移到CPU并按完整恢复继续Task7，不重跑Task6。CPU preflight核100键CP与恢复model精确、全部状态字段及100轮history；GPU实际恢复和自己的Task7准入待最新快照确认。原恢复文件未额外复制，不因紧张容量伪称保全额外payload副本；原model-only Task6 CP永久保留，完成history/原失败/hash和源代码均保留。

新root=/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02，controller=20530，合同56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71，65冻结项。旧根不再是活跃目标。科学参数/输入/初始化/预算不改，未重试科学负结果。
# 实现与边界

新runtime、train、controller为完整序列，B/C/N/V公式沿用已审查原型和原控制组；每组build只在组起点调用一次，从官方预训练载入100键，非Stage8继续。stage setup使用原优化器/调度器，保留组内随机流。只读复用原阶段manifest，避免最终buffer人口污染早期阶段；辅助扩大至当时所有合法旧窗口，教师为自身上一阶段。

updates.py的梯度诊断改为保存并比较p.grad，允许None或已有零tensor，不能用allNone判断无副作用。每个实际旧阶段都做有历史梯度的测量开关对照。原错误源和9epoch失败证据不覆盖。新程序尚须真实准入，不以AST或旧准入冒充PASS。

闭环扩为全部三角矩阵800episodes，无额外环境准入、无科学负结果重试。存储复用原输入payload，自己模型全部重新训练。Stage9硬链接完整恢复，Stage6–8存模型；每组最后恢复完整保留。经授权删除旧中间恢复副本后，相关历史中断状态不再能重新演练，但旧最终模型/最终恢复/独立验收和hash记录完整保留，该限制不能隐瞒。
