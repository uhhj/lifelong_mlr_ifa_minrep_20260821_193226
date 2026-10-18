# RA-Coreset GOAL seed42：已启动，PARTIAL

观察时间：2026-10-18T23:35:47.720504+08:00。正式supervisor PID48910、训练worker48924，Task6已完成38/100epoch、3800/76100优化步；正式闭环0/200，尚无最终结果。GPU RTX3090，系统盘约19.05 GiB可用，冻结hash PASS。

用户要求“实验协议和上次一样，以用户为准”优先于附件：仅新增RA选样，保留原文件排序/concatshuffle/sampler0和每stage global动作RNG，不启用stable-slot或E1独立噪声。复用已有Coreset/Random指标，不重训重评基线。新组400epoch76100步4ep99模型200闭环20vectorattempt；stateTrue、seed42、BC+.1meanIFA、gamma.7、batch10、warmup500逐步线性不变。

|任务|窗口数|配额|随机锚点|实际最少交换|原半径|RA半径|
|---|---:|---:|---:|---:|---:|---:|
|6|997|499|249|127|0.278769|0.297022|
|7|889|445|222|108|0.261264|0.270088|
|8|904|452|226|110|0.290898|0.304558|
|9|1969|985|492|254|0.241957|0.251222|

Task6–8共345条旧成员差异；Task9的254条在训练结束后入库，不影响已完成模型。半径变大符合RA设计：必须纳入随机锚点时最小化几何损害，不承诺不退化；这些数值不是闭环成功结果。

已完成：原附件哈希校验；8项合成CPU测试；真实4759窗口载荷与身份校验；四任务原Coreset及Random完整有序名单复现；每次所有可删除成员独立最近距离复算；锚点全包含、交换次数最少、配额/唯一性、全局RNG保护；恰好4次隔离GPU原BC+IFA更新及完整恢复一致性、0环境/0策略rollout；正式从官方权重重新seed。164冻结源码和32份启动轻量包本地复核PASS。实际模型/损失/原加载器/Distance/评估计算未变；train/evaluate仅命名空间、arm及父路径变量不同。

部署两次前置失败均已保留：Python3.8路径API兼容、安装器白名单遗漏LICENSE/txt/sbatch。修复的是安装工具，未进入训练或环境，也未修改已冻结科学合同。之后software/prepare/offline/freeze/preflight均exit0，正式进程当前没有失败退出码。

实际命令：

```sh
python3 /tmp/ra_goal_deploy_20261018_v3.py
bash /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/admission.sh
source /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh
/data/mlr_ifa/env/bin/python /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/launch_once.py
```

admission依次8软件、prepare、真实select_offline、freeze、torchrun单GPU preflight。正式supervise后台自动执行`torchrun -m coreset_exp.train ra_coreset`、独立`python -m coreset_exp.evaluate ra_coreset`、`python -m ra_exp.audit`、退出验收。不手动增加rollout或重复启动。

按本轮Task6实测约2.76秒/100batch及上轮同机train2328秒/evaluate4318秒，观察时预计剩余约105–135分钟（训练约35–45分钟、评估65–85分钟及验收）。预计北京时间10月19日01:20–01:50；非完成承诺。每两小时监控ra-coreset-goal-seed42，仅实质进展/恢复/失败/完成通知，完整交付后取消。

门槛保留RA附件三条件：AUC>78.75%，FWT>=81.25%，OldFinal>=68.333333%，且全工程PASS。Random同时报告（AUC80.9375%）。按附件no_auto_extend，先交付首轮门槛，再决定后续套件/seed，不靠离线半径或训练loss判成功。

限制：GOAL42已用于设计，为探索性筛选；未隔离共同样本重新排序、实际逐轨迹动作噪声或IFA间接变化；state原预训练元数据仍unknown，父调度与ep99规则是工程补全。不能宣称纯记忆成员因果效果。训练结束而评估/验收未齐时不能COMPLETED。

本地路径execution/20261018_ra_coreset_goal_seed42；服务器源码/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/source，产物/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018。发布仅代码/协议/脱敏元数据/日志/哈希；不含权重、latent、buffer、演示、真实动作或凭据。
