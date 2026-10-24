# 实施说明

独立新包 caet_em_exp；原ALIGN/RA源码仅只读导入原生模型、人口/IFA、原正式rollout。新模型同时覆盖训练和完整部署读出。原执行目录不修改。

已完成源实现，engineering04自己的codec GPU准入实际通过且训练启动；三组策略准入与完整闭环仍待实际执行。AST通过不是GPU或科学PASS。完整3组有限队列无性能条件分支，无额外seed。工程失败版本及精确改动与成本均保留，不覆盖冻结源码。

本地首次AST发现transfer报告括号缺失，冻结前已修正，0服务器执行/0更新；无需恢复训练。所有实际服务器失败另存原始ATTEMPT与日志。


工程05：诊断按训练manifest重建历史阶段人口，修复retain后扩大buffer的KeyError。原训练数据构造不改；仅CP加载增添两份SHA+payload fingerprint严格继承白名单。新controller跳过已完整codec/B0 Task6/7，先真实恢复态GPU roundtrip及本版本B0准入，再继续诊断7。旧输出复用、旧失败和源保留，新的合同完整绑定继承资产；详见ENGINEERING_REPAIR_05与RESUME_MANIFEST。


2026-10-23 23:15 CST：新增纯本地audit_evaluation_progress.py，实际对snapshot_20261023_230628通过四阶段诊断manifest/CP/codec绑定及已记录formal attempt/episode一致性审计；不计算部分性能，不替代最终CPU严格CP及初态文件字节审计。冻结1875源码未变，B0完整训练结束，按首批实测和用户新规则下次检查90分钟后。


2026-10-24 01:45 CST：新增外置capacity_readonly.py，仅两次stat稳定私有文件st_blocks计算剩余新增界；symlink与hardlink排除，额外保全恢复只在实际free中计一次。实际远端只读及本地算术PASS，未修改冻结科学源。


2026-10-24 recovery06：服务器连接恢复后发现原controller/训练进程消失，无退出标记。冻结1875项保持；独立版本化接续supervisor从B2 Task8完整epoch90使用原train --resume，无科学源码修改。原状态、日志、EVENTS、准入报告及365436131bytes恢复态私有保全；先自己的原准入及实际保存状态下一步精确重放，再继续未完成阶段。完整Task6/7和B0/B1不重跑，未知未提交epoch开销单列。最终外置验收追加执行合同及前90轮精确绑定，尚待最终实际验收。


2026-10-24 14:12 外置检查修复与收尾准备：新cell尚无完整vector时允许未生成trials，仍要求无完成事件且不在snapshot/download文件表；旧源/失败及真实复算SHA保留于snapshot_20261024_140806。新增本地B2完整训练/Task8恢复前缀与教师绑定审计实际PASS。describe_final追加已验收B0 CPU动作诊断成本SHA绑定，仅AST通过，未对未完队列运行最终验收。服务器源/参数/进程无修改。


2026-10-24最终验收：final_readonly、audit_final_local、describe_final首次真实运行均PASS。新增final_payload_binding_readonly实际CPU核3组全RA/sidecar/三教师缓存及mask，无前向。close_final本地schema/index和路径问题保留两次失败源并修复，完整矩阵/成本/闭合真实PASS。无服务器改动或重训重评。最终目录final_CAET_EM_20261024_153414。
