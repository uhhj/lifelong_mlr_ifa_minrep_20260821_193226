# 三方案与 LIBERO-50 持续执行交接

本任务尚未完成，不能只轮询而放弃剩余实现。用户要求持续工作、自动选推荐方案处理工程问题；如服务器无LIBERO50数据，自动取得。不得伪造实验完成、扩大科学预算、重试正常失败或覆盖失败证据。

## 最新明确授权

停止旧RCTR A2（已停止，143，76完成/77启动，零训练）；依次RCTR、CAET、COVR，均RA+IFA。各OBJECT42先跑；AUC严格高于同seed RA且工程通过才补41、43。三个OBJECT等权平均AUC严格高于对应RA平均才跑该方法GOAL41、42、43，之后下一方法。无2/3或OldFinal附加门槛。全部方法适用分支完成后RA+IFA LIBERO50 41、42、43。用户允许自主选择，无需再问常规实现/下载/工程恢复选择；科学改变必须如实单独记录，不能改阈值冒充原协议。

## 当前真实状态（2026-10-21）

唯一正式worker：RCTR OBJECT42。
W=/data/mlr_ifa/experiment_workspaces/RCTR_OBJECT_seed42_full_20261021
O=/data/mlr_ifa/experiments/RCTR_OBJECT_seed42_full_20261021
supervisor PID46513，pipeline PID46522。source W/env.sh。
GPU准入第二次exit0，28工程步、0optimizer、0正式；首次exit1是缺assets链接，第一个环境步前，证据保留。LAUNCH_ACCEPTANCE与本地audit_admission_local.py PASS。source176文件冻结；不要修改已冻结RCTR source或原CONTRACT。
最新已完成Task6 100epoch/15700更新，进入collect6；Task6逐张量和初始化RNG同RA父通过才会进入采集。三个采集阶段在各任务ep99之后，最多1368探索+144确认，前缀与工程重复另计。每原训练batch仍只一步optimizer，最终105500更新、200原协议正式episode/20vector。
pipeline.py会按train6/collect6/train7/collect7/train8/collect8/train9/evaluate/audit执行；supervise.sh待pipeline结束后final_process。新生成正式任务未完，不能说全部方案完成。

## 条件队列真实部署状态（更新）

Q=/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020，controller PID46681。9个OBJECT入口+6个RCTR/COVR GOAL入口已登记，REGISTRY15项。旧OBJECT源/entry/deployment hash完整保留。CAET/COVR实现已在204文件文本bundle中，真实GPU准入只在条件触发时执行，CPU11项已PASS不当GPU证据。不要重跑prepare或改已登记共享文件SHA；新修复用明确版本、保留原失败合同。
queue_controller.py→branch_gate.py→queue_verifier.py独立验收全记录再判精确AUC。CAET GOAL真实探针发现stage7正对为0（41/42/43全部），因此没有注册；其他stage8/9有正对。阈值不能放宽，不静默跳过适用方法。LIBERO50 worker/验收adapter尚需实现，当前controller明确拒绝RA_IFA adapter缺失；不应声称整套已部署。

## 资产与下载

SSH root@jq1.9gpu.com:11920已有授权。凭据仅内存/getpass，不写文件日志。可使用已有PTY JSON桥（当前session27015，脚本execution/20261020_epr_stageA/ssh_session.py）；exec/put/get元数据，禁止输出凭据。
父OBJECT RA /data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed{s}_20261019；对应已完成RA源码在RA_OBJECT_conditional_20261019/seed{s}/source。原基线只读不重跑。
本地source/为已部署RCTR源码，PARENT_SOURCE_BUNDLE.json为原真实RA文本源，hash来源PARENT_SOURCE_HASHES。
官方LIBERO镜像yifengzhu-hf/LIBERO-datasets，固定revision f13aa24a3da8c43c7225569f28c562979fa0e35a。download_libero50.py后台PID25160在下载50厨房任务约39GiB，逐文件LFS SHA与大小校验，本地local_assets/libero50_official，绝不发布数据。检查实际PID和partial/status，有限网络失败安全续传，不重复进程；本地D盘约84GiB初始余量，远端/data25GiB/root13GiB初始余量，所以先本地取得，再设计分批转移/合法预处理。
作者公开目录目前只找到OBJECT和GOAL预训练权重，LIBERO50权重未找到。继续查官方资产；必要时依用户自主选择授权准备忠实预训练复现，但必须公开来源/实现差异，不能用OBJECT权重充当50官方权重。论文50协议25预训练任务各50演示+5阶段各5任务各10演示，100epoch、batch10、gamma.1。任务划分可核验官方LOTUS LIBERO_KITCHEN_EXP25，asset_sources/lotus_benchmark.py有索引；还需严格验证与作者协议。仅下载全部50厨房数据不表示划分/权重已齐全。

## 恢复与发布

确认原进程已退出、合同哈希一致、保存错误后才恢复。完整epoch恢复模型/optimizer/scheduler/grads/模式/allRNG/cursor；存储预提交清单幂等，不重复入库。RCTR collection每key committed JSON是完成权威；不能重试已提交正常失败，技术恢复重采同固定prefix必须和原物理/观察hash一致并另记全部attempt。正式global RNG评估部分完成不能盲重启。
保持source冻结；必要工程修复用有因果说明的外置适配或明确版本和恢复合同，绝不重贴原标签。
每轮检查/审计/实验发布到publish_mlr_branch远端MLR，运行publish.py，审查代码和轻量元数据，排除文本源bundle/asset_sources/权重/原actions/latent/buffer/状态/RNG二进制。最近已核验远端commit5df28b4f242a7bc9ec48c4855432ecdc5ce286c9。后续新草稿尚未发布。receipt本地保留；不能称出版完成即实验完成。

## 2026-10-21最新补充

以REPORT.md和RUN_STATUS.json时间为准：RCTR collect6已64 committed，0正式。服务器acquire_libero50.sh PID46818已接管下载，LIBERO50_SERVER_STATUS/CURRENT和libero50_acquisition.exit为权威；本地PID25160已停止，文件保留。无损打包预检PASS，原数组/属性全等才删除自身暂存raw。当前3/50已验证。GOAL registry首次换行SHA失败、探针前两次缺env导入失败保留，v2注册/v3探针通过。不得把旧交接中本地下载和“队列未部署”描述当最新状态。
