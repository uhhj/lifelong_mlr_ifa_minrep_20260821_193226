# 实现与工程边界

新队列Q=/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023，controller95924；六入口TACR41→条件42/43→独立TACRST41→条件42/43。已冻结1462源码/调度文件，合同3403553fc5bd26db0045a80c9e7db156fde52f0f633bd85833d520a7d397da2b。旧ALIGN→GOAL合同8c35a51e3bf0d4c3c7a5b55b642a88d6b2dca861cd3c7099fd31fbc94e8e21ca不改。新controller只持自己队列锁等待旧exit0，随后核旧1209hash、所有实际完成receipt/GPU空闲，再取得旧queue锁；无kill、无覆盖旧合同。

model.py定义20键学生，其中部署14键R/U/gate，E/D训练专用。observe.py沿用主PaperMLR骨架，控制与表征参数无重叠、同主batch一次联合反向/双优化器/双scheduler/三组裁剪；train.py保存原100键主CP、独立训练分支与部署分支、恢复状态。evaluate.py保留原eval_metric的coreset_exp.evaluate CONTEXT/事件路径；严格加载100键后保存恢复RNG构建并加载14键ResidualHead，记录branchSHA，原闭环Normal采样与时间/初态不变。audit.py在CPU严格加载主/部署/学生，并核训练部署融合一致；audit_mechanism.py复算原shuffle+sampler、逐步mask/效果曝光/游标/固定教师与双更新账目。teacher_keys.py冻结上一阶段T/R/E、缓存实际旧RA合法成员、绑定主CP与学生CP/缓存及样本SHA。GPU准入的ST激活使用显式标注的构造键验证不可变及游标恢复；实际图、实际阶段教师另审计，不用构造样本冒充真实迁移。

零训练三seed图真实重建：Task8/9支持41=156/207、42=152/205、43=180/200；Task6/7均0。夹爪sign条件在互邻和剪枝前加入；不能套用旧178/206，也不能由覆盖宣称效果。actual_cpu_20261023_001129对实际BriefPolicy Temporal/GMM作CPU自动微分，BC进入T/G/R/U/gate，IFA只T，effect只R/E/D，ST只R/E；部署融合精确。cpu_checks_20261023_000439另核初始原GMM参数/样本精确、U可学、R初始BC梯度为0且有独立效果梯度、14/20键、两参数集合无交集。均0优化器/0环境/0CUDA/0服务器写入，不替代待执行的每run8次真实GPU优化器准入。

只读尝试保留：cpu_checks_20261022_235858连接读取超时，后续cpu_probe核无诊断残留；actual_cpu_20261023_001015因遗漏CORESET_OUT环境变量在导入阶段退出1，修正外置探针环境后001129通过。前者不推断为科学失败，后者无模型构建/更新；均未触碰正常ALIGN评估。

原完整ALIGN41输出实测1,314,732,242字节、当时ALIGN42为1,308,784,244。每新run预留1.6GiB包括新分支、教师与日志余量；三run普通system、三rundata root。launch_20261023_001410核system普通7,979,462,656字节，对新三run5,153,960,754＋1GiB；data root15,000,326,144字节，对旧待运行四run＋新三run12,025,908,426＋2GiB通过。不是全程最终容量保证，controller每run重新验全剩余上界；旧资产删除0。数据特征只读链接，原RA buffer新复制不改采样排序。

包装package_final只列JSON/JSONL/exit白名单；包括部署SHA与teacher metadata，不含pth/缓存/图标签tensor/恢复payload。resume并行全SHA后独立audit_final重算全矩阵与账目/receipt，aggregate_local为OBJECT每variant三seed等权与样本SD(ddof1)，扩展条件只看各自41>701/960。新最终路径尚未有完成实验，必须真实验收；外置包装错误修复不重训重评、不更改冻结源码。用户允许工程修复，不允许科学阈值随结果调整。

import_readonly_20261023_002022：六入口各15个实际活动模块全部导入PASS，自己的variant/seed匹配，torch CUDA未初始化、无模型前向/优化器/环境/输出创建，冻结1462hash不变。此前两次外置探针分别因空CUDA环境触发LIBERO强制EGL设备解析、以及不相容的PYOPENGL_PLATFORM而在导入期退出1，原ATTEMPT均保留；改为worker同款EGL设备环境后通过，未改变任何冻结代码/正常评估。

同一align-object-seed-goal监控已更新覆盖原A与新N。首次无RRULE前缀更新被工具保存120分钟，复核后用工具提交带前缀的30分钟规则成功；AUTOMATION_UPDATE_RECEIPT记录实际保存值与提示SHA，不手改调度文件、不新建重复任务。


2026-10-23 01:25 CST：snapshot_readonly_20261023_012140/LOCAL_AUDIT PASS，新95924仍WAITING、1462hash不变、六输出不存在，无GPU准入。旧ALIGN42已在A/final_OBJECT_42_20261023_012252完整本地独立PASS，AUC83.020833%对同seedRA71.5625%；旧43已自动GPU准入并训练中，本N不越过旧GOAL条件流程。capacity_readonly_20261023_012323按剩余新增分配计当前43已写稳定私有1.009GB的信用，data root13.990GB，对最坏剩余11.017GB＋2GiB后仍余0.826GB；system普通7.979GB仍余1.752GB。不能拿部署时全额四run界重复扣当前已耗的空间而误报满盘；此仍是规划界，不是最终容量验收。

外置resume_light_download.py发现原复制工具仅允许/data元数据路径，已加明确的/root/mlr_ifa/experiments/前缀以匹配冻结TACR41/43与TACRST42输出；仍只.json/.jsonl/.exit、包目录内路径、<8MiB、manifest全SHA，不下载pth或payload。仅外置包装修改/语法验证，无服务器源码或合同修改；实际新组最终下载仍待真实验收。N无新性能结果或可用吞吐ETA。

2026-10-23T10:56:37.860486+08:00：首次正式部署进度由外置deployment_checkpoint_readonly.py在独立CPU进程逐阶段读取主/部署/学生CP，仅结构/参数/元数据检查无forward；audit_deployment_progress.py本地将当前原始episode/attempt/cell的100+14键加载记录及SHA绑定这份证据。主模型CPU未实例化或strict加载，未访问的阶段严格加载仍待正式/最终原审计。1462冻结源不改、探针0CUDA/前向/优化器/环境/服务器写入，不发布CP payload。实际证据deployment_checkpoint_readonly_20261023_105439和snapshot_readonly_20261023_105138/DEPLOYMENT_PROGRESS_AUDIT.json通过，不等于完整run最终验收。
