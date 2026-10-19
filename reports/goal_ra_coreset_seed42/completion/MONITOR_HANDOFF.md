# RA-Coreset GOAL seed42，原协议运行交接

先读PROTOCOL.md、ACTUAL_SPEC.yaml、RUN_STATUS.json、LOCAL_LAUNCH_ACCEPTANCE.json（若已产生）、launch_evidence/CONTRACT.json和最新恢复记录。用户最新要求实验协议与上轮一致，覆盖包内stable-slot和E1建议；本轮仅RA选样改变。旧基线/SC-v1/SC-v2及其他实验保持只读。

SSH root@jq1.9gpu.com:11920已有授权；凭据只内存/getpass，不保存文件日志。W=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018，O=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018，P=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017。source W/env.sh，用/data/mlr_ifa/env/bin/python。新GOAL产物沿用系统盘授权。先跑W/monitor_snapshot.py并核验真实进程、GPU、磁盘、退出码、epochs/recovery/storage、日志与正式trials/attempt，检查冻结source/config/modal/input/feature hash，禁止重复启动。

首轮仅seed42、ra_coreset，官方GOAL权重、同父stateTrue缓存、空buffer，任务6–9各100完整epoch/batch10共400epoch76100step。原BC+.1meanIFA/gamma.7/warmup500逐step线性、原路径排序concatshuffle/sampler0、网络跨任务继承。q499/445/452/985，随机锚点249/222/226/492来自同seed原Random完整q名单前半。原S0与Random名单须逐有序ID复现；缺失锚点按原顺序加入，删非锚点使原最大距离、均值、ID依次最小，1e-12容差。恰好|A-S0|次交换，成员/顺序必须与冻结offline一致，不添加SC构成/几何硬约束，不改距离/比例/seed，不重选旧任务。IFA用实际新记忆原算法计算。

8合成测试只软件，真实4759窗口/原载荷哈希/独立所有删除候选验证/选择不消耗全局RNG必须PASS。恰好4次隔离GPU原BC+IFA更新与恢复测试、0环境。通过后不重复预检。首次Task6实际模型tensor和初始化全RNG与父相同，旧模型不重训。正式重新初始化RNG，不接预检状态。

评估仅4ep99最终模型200条20vector，原0–19初态、10lane/env42,43、600步、5零动作、每stage全局RNG42、argmax分量Normal；不启用E1、不重评Random/Coreset、不加680/9000。模型阶段/前端/modal合同一致。不能将同配置称实际逐轨迹噪声相同。

后台supervise.sh安排train→evaluate→audit→退出验收；训练结束不是完成。完成要求400epoch76100step4CP严格加载200条20attempt、完整矩阵与精确AUC/FWT/NBT/OldFinal/FinalAll、Task6tensor、配额/锚点/最少交换/离线identity/输入hash、pipeline和supervisor退出0无残留，PROCESS_EXIT_ACCEPTANCE PASS。原失败退出码保留。

恢复先确认进程停、合同输入一致、保存失败证据；只从本arm完整epoch恢复model/optimizer/scheduler/grads/modes/allRNG/cursor；按预提交清单幂等复制、不重抽或重复计步入库。部分global RNG评估不可盲重启，正常失败不重试；技术恢复须保持完整attempt与原RNG环境顺序，不能保证则BLOCKED。冻结代码不悄改/重贴合同。只读汇总工程故障可外置修复不重跑环境。

实际门槛：AUC>63/80，FWT>=13/16，OldFinal>=41/60，全部工程PASS；Random参考AUC259/320。沿用附件no_auto_extend，先完整交付首轮和门槛，不擅自启动OBJECT或扩seed。若用户随后批准后续，必须新suite/seed合同。原科学设计不靠offline覆盖或loss决定有效。

W/package_light.py只打包元数据并删除绑定last_action字段；不要下载checkpoint/RNG二进制/latent/buffer/演示。验证PACKAGE_MANIFEST，本地独立复算。每次检查及实验轮次按AGENTS发布到publish_mlr_branch的MLR分支，审查凭据/扩展名/体积/hash，仅当前产物，禁止force push。正常保存快照和实测ETA结束，不持续等待。披露探索性单seed、顺序改变、global动作RNG、pretrain state unknown、父调度补全及IFA间接变化。

部署时两次安装器错误均发生在选样/训练前：Python3.8不支持is_relative_to、扩展名白名单遗漏源LICENSE/txt/sbatch；保留W/INSTALL_FAILURE_1.json和2.json与/tmp各版本安装器。修复仅安装工程，无正式训练/环境调用，不是运行中断或模型恢复。

正式supervisor PID48910，pipeline48915，训练worker48924于2026-10-18 23:33 CST启动。8软件、真实离线及4GPU更新恢复已PASS。本地verify_launch复核32份轻量证据、164源文件通过。继承用户每两小时监控要求，自动化ra-coreset-goal-seed42已启用；首轮验收交付后删除。冻结spec中的原设计阶段create_automation_now:false不修改，实际监控授权和本条操作记录单列。

2026-10-19 最终验收完成：服务器400epoch/76100step/4CP/200episode/20attempt、所有exit0，冻结hash和本地精确复算PASS。AUC82.5/FWT85/NBT5.555556/OldFinal75/FinalAll80，三条件门槛通过；按no_auto_extend未启动其他seed。自动化已删除。详见FINAL_REPORT.md、LOCAL_FINAL_ACCEPTANCE.json与AUTOMATION_DELETION.json。
