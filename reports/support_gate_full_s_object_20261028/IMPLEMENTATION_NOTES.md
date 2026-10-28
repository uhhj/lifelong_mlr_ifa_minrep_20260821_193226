# 完整 S 实现与来源

build_local.py 在新目录组合已验收原完整 S OBJECT seed41/42/43 与已运行机制门控实现。runtime、初始化、采集、评估、旧缓存读取、losses、synthetic_checks 对各 seed 原文件文本精确不变。逐文件原SHA/新SHA及diff位于UPLOAD_REVIEW和review。三seed各自PARENT_BINDING来自本轮真实只读preflight，不能互套数据mask。

support_memory读取本轮自己的成功采集缓存并派生q，不使用机制旧BINDING。fork_tools的stage由真实Dataset提供；gate_admission每stage指定自己的stage。Task6按原完整S的4更新准入；Task7–9每stage真实29更新准入，完整状态恢复后才训练。新train仅改变支持加权/双候选选择及其记账，阶段采集和正式评估未改变。

每seed195300实际训练调用=105500接受+89800额外C候选。每seed门控9100块（7/8/9各2300/3100/3700），三seed27300块。接受Q时才计其4槽/步；拒绝候选的算力计入physical但其损失不计入接受轨迹totals。每epoch保存完整state，stage9硬链接最后恢复文件。新准入全部273调用/123额外autograd，方向诊断最多108autograd，环境采集120、正式600。不得把内部门控C/Q误当独立重跑机制组。

仅外置监控/验收包装可以版本化修复，不修改63个冻结项；最终独立CPU包装尚须从原完整S适配，本轮只有静态及实际GPU首阶段准入，不能宣称CPU最终通过。final_readonly只可在自然controller.exit0且无自身进程后执行。预检和部署使用现有credential worker，秘密仅内存；新目录不含会话目录。
