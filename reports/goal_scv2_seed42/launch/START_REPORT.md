# SC-v2 GOAL seed42 启动报告（PARTIAL）

2026-10-18 20:23 CST真实快照：已完成54/400 epoch，当前Task6第54/100轮完成（epoch_index=53），5400/76100 optimizer step；GPU RTX3090正常，0正式闭环、0技术异常；train/evaluate/audit/pipeline退出码尚未生成。尚未完成，不存在SC-v2成功率结论。

离线全部通过：Task6–9交换5/2/4/6条，独立监督495/426/449/981均保持，原示范及终点支持保留，最大/平均几何距离不退化，E逐次严格下降；全4759窗口和原S0顺序真实核验，4781输入前后hash不变，独立全搜索复核17次选择及终止。Task6–8实际改变11/1396条旧记忆，属于小幅干预。

目标端5软件测试通过；恰好4次隔离GPU BC+IFA原生更新/恢复通过，0环境/0策略rollout，实际storage接口逐项复现离线选择并保持全RNG。启动验收本地核对31个轻量包文件及163个冻结源文件，PASS。模型、训练器、损失、原evaluate/eval_metric字节未改，仅选择及对应准备/验收改变。官方权重重新seed42起步、原stateTrue缓存、独立空buffer，100完整epoch×4、batch10、76100步、4ep99CP。

用户最新选择“按上次实验的计划来”已落实为上一轮SC原协议：只新评SC-v2共200条，原0–19初态、每stage global RNG、原GMM动作分布，复用同seed旧Random/Coreset；不启动600条逐轨迹RNG重评。实际跨模型动作随机流仍可能不同，不宣称逐轨迹噪声隔离。扩展门槛固定AUC>78.75%且OldFinal>=68.33333333333333%，工程全部验收通过后才扩seed43/44。

实际部署/启动命令：

```sh
/data/mlr_ifa/env/bin/python /tmp/deploy_scv2_goal_20261018.py
source /root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018/env.sh
/data/mlr_ifa/env/bin/python -m unittest -v sc_exp.test_selection
/data/mlr_ifa/env/bin/python -u -m sc_exp.prepare
/data/mlr_ifa/env/bin/python -m sc_exp.freeze
/data/mlr_ifa/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 -m sc_exp.preflight
/data/mlr_ifa/env/bin/python /tmp/launch_scv2_goal_20261018.py
```

上述工具调用退出0；后台supervisor PID43881/pipeline PID43886/训练进程PID43895继续运行。正式命令由pipeline记录为`python -m torch.distributed.run --standalone --nproc_per_node=1 -m coreset_exp.train sc_coreset`，完成后依次原evaluate、SC-v2 audit及进程退出验收。内部arm名sc_coreset未变，版本以独立SCv2目录和合同区分。

新产物：`/root/mlr_ifa/experiments/SCv2_Coreset_GOAL_seed42_20261018`；原基线只读。冻结合同SHA256 `f7bd783a723545173d53c29fd60a7805890836c99f8c3f50874bf242ec3e876d`。预计训练约40分钟，评估约70–85分钟，结合原同机测量预计22:15–22:40完成，非保证时间。自动化`sc-v2-goal-seed42`每两小时检查，临近结束可缩短；全部授权工作交付后取消。

本地离线复算：`python -X utf8 execution/20261018_scv2_offline/recompute.py`；启动验收：`python -X utf8 execution/20261018_scv2_goal_seed42/verify_launch.py`。按AGENTS只推代码、报告、脱敏metadata和hash，不推权重/载荷/原动作。
