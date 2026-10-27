# 支持度门控机制实验 → 最新完整S OBJECT41/42/43

同一监控ID为 `s-object-seed`（此ID由应用分配，名称“支持度门控机制与完整S OBJECT三seed”），不要与已删除的旧S OBJECT监控混淆。更新本监控的间隔和最新交接，不建立重复监控。

用户最新澄清具有最高优先级：**先完成机制实验全部既定对照；随后完整序列只跑最新完整S（逐任务支持度+10步C参照门控）的OBJECT seed41、42、43。** 不是取消机制对照。用户要求自主决定最推荐选项、解决工程障碍直到全部实验验收发布，不自动增加其他科学实验。新任务后续未完成前不能删除整个监控。

先完整读本目录PROTOCOL、IMPLEMENTATION_NOTES、REPORT、RUN_STATUS、ACTIVE_ROOT和最新证据。机制唯一root `/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1`，controller29838，合同`9b53df5a8fbe894e7b27887462b7b25b4e8477a9ad12820deb8df162632fb978`，56冻结项。禁止重跑bootstrap/build_local/deploy/controller或重训完成阶段。父所有旧实验结束，只读复用，不恢复。

实际23:56:22CST观察check_readonly_20261027_235621/LOCAL_AUDIT PASS_PROGRESS_NOT_FINAL：GOAL41分叉C完成53/100epoch9805接受更新；自己的29隔离GPU调用及11autograd准入PASS，0环境。lambda0=原C、q1=原S、10步C精确重放、完整保存恢复、测量不改变更新、探针不改变全state及1–8长度实际通过。其余组仍须自己的准入，不能套PASS。controller29838/GPU29855正常，合同与父源不变，system留1GiB后余363835392B，data抵扣8GiB总峰值且留2GiB后余34610573312B，无清理付费。最近10轮均8.533378秒，剩47轮约6.68448min纯训练，加保存/准备另计；下一10min在本run完成附近检查，再根据当前整run或全评估真实剩余时间更新同一监控，不固定高频。

本轮17个单阶段run：GOAL41/42/43各C/S/Q/G，OBJECT41 C/S/Q/G/U。GOAL从已有S7只训8，OBJECT从S8只训9；同CP/缓存/主batch和新阶段共同完整state。旧中间CP没有全RNG，显式新阶段seed重置和优化器重建，不称原轨迹精确续训。GOAL41/43 q全1，Q与S完整最终tensor必须相等；GOAL42 q都.9，所以其对照不能区分任务分配与全局缩小。OBJECT41 q=.1/1/1，U=.7。

预算1700epoch403000接受主更新，G额外C候选91700，总实际训练优化器494700；自己的隔离29*17=493调用/187额外autograd，正式梯度最多204autograd，探针前向另外计数。无训练采集。全部训练后1120正式112vector56cell，各run评本阶段所有已见任务0–19，原600步5zero和配对Normal噪声。门控每10步、epoch尾短块，同state分别C/Q，当前训练64合法窗口BC和执行专家cost均<=C且旧固定8窗口/任务支持加权保持严格<C才选Q，否则C。两个epsilon0；不扫参/选测试最佳，所有候选完整state恢复。q乘保持回归及route且分母固定4，不重归一化。

凭据worker4572通过用户授权Windows Credential Manager，先读.session_control/status.json，不重启活跃worker、不打印凭据；确退出可隐藏启动credential_session.py。`queue_remote.py check_readonly.py`后读对应result实际返回目录，不猜秒数；`audit_progress.py SNAPSHOT`→`record_progress.py SNAPSHOT`。check包含容量stat和进程/hash。queue result输出较长会截断，实际EVIDENCE完整，直接读返回目录文件。记录生成器当前硬编码首次deployment路径，工程版本变化先外置适配，不覆盖原证据。

故障先核相关进程停止、冻结hash、完整epoch恢复、原失败/候选及环境attempt；版本化工程修，不覆盖科学冻结源。禁止盲重启部分G块/正式评估，不能丢弃已耗候选成本，不能重复完成epoch。正常负结果不重试。用户授权处理工程错误，不等于修改损失/阈值/预算。当前无远程工程失败；只有本地辅助编辑PowerShell引用解析错误，未修改目标，已记录。

全部自然controller.exit0且无自身进程后`queue_remote.py final_readonly.py`→`audit_final_local.py FINAL`→`describe_final.py FINAL`。外置final wrappers只是AST/静态，尚须真实17CPU CP、完整恢复、共同初态、父manifest/RA/mask、缓存teacher重放、1120初态字节noise、逐块决策及标量验收。外置包装错误保留原错误修复，不重训重评。机制解释必须结合新旧任务闭环、SF/FS、当前BC/执行误差/保持梯度方向和门控接受率，不能只凭自动统计。支持度不称成功概率，探针是训练数据不是独立验证。

**机制接受和分析发布后，继续用户已授权的最新完整S OBJECT41/42/43。** 详见IMPLEMENTATION_NOTES冻结建议：仅新S一个方法；三个seed各官方OBJECT初始化、Task6–9各100epoch105500接受步；Task6无保持，7–9使用本轮q和门控，原S每task10次训练采集固定初态20–29/成功缓存quota/不补采，三seed120训练+600正式0–19。每seed内部C候选额外89800，三seed实际585900训练优化器，准入另记。不新训C/原S/Q/U；旧结果只读参考，不称等交互/等算力。不要根据机制测试调块长/epsilon/损失；不因负结果自动加实验。新独立目录/root/合同、自己GPU准入/CPU最终验收。当前完整阶段代码尚未构建或部署，监控必须自主完成此工作；不能把机制阶段完成当作用户最终目标完成。

每轮经本目录publish.py按根AGENTS审核轻量code/report/hash/scalars/outcomes、串行stage非force推MLR，核remoteSHA/localreceipt；OpenSSL/HTTP1.1仍验证证书。禁模型/CP/action/state/features/buffer/RNG/凭据/.session_control/未审bundle。reference_source不发布，原源hash和差异保留。只实质变化/失败/完成/需行动通知，健康无变化静默。全部机制+后续三seed独立验收、报告解释及发布后才删除监控。
