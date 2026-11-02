## 最新2026-11-02 13:25 用户要求检测速度

实际check_readonly_20261102_132538审计PASS，132027全部epoch/门控/诊断前缀精确保留，135科学/5旧工程/8恢复工程冻结项不变。L005_41 Task6/7/8各100轮，共300/400轮69300接受122900完整轮生产，62自身隔离30准入autograd、30采集，正式0；其他五run未开始。controller2560/GPU2566正常，Task8训练采集收尾，Task9尚未实际准入，不能套PASS。

恢复后Task8完整40轮均53.7142秒、近10轮55.3703秒（54.05–58.05）；暂停前同Task8近10轮268.5262秒。观测吞吐4.85倍、耗时减少79.38%。宿主机/GPU及设备快照同时变化，不能归因单一因素，不能据此说旧GPU硬件损坏或λ导致慢速。未来Task9未测，按362/309批次条件外推纯计算约108.1分钟，计保存/准入/额外探针/最后10采集，当前整run规划120–160分钟。下一120分钟在当前run结束较早侧复查，不固定高频；其他五run和全部正式未测，不给可靠全队列完成日期。

新工程及硬件授权、恢复成本与全部科学边界不变。原260轮和全部4167门控前缀保留；8137/47工程诊断、原中断140已记账候选+UNKNOWN额外成本单列。最终只认engineering_resume_3090_20261102_v1/controller.exit0及无自身进程，原根/旧工程exit不得伪造。未来stage必须自己的准入/BRANCH_COPY_EQ；未来run初始tensor/RNG相同，Task6跨GPU仅记录实际比较不预称相等。详见HARDWARE_AUTHORIZATION_ADDENDUM_20261102及RECOVERY_REPORT_20261102，硬件混杂与开发集/预训练来源限制必须进入最终报告。最终CPU仍待真实全队列完成。现worker3700，先核status/PID，禁止重复dispatch/controller/build或已完成轮。

本段覆盖13:20旧295轮、125分钟间隔，科学仍PARTIAL。

# OBJECT 最新S固定λ_R六run：当前交接

实际观察 2026-11-02T13:25:38.910742+08:00；check_readonly_20261102_132538/LOCAL_AUDIT=PASS_PROGRESS_NOT_FINAL。

唯一root `/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1`，controller2560，合同`6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e`，135冻结项。唯一部署deploy_remote_20261101_152959。禁止重复compose/freeze/deploy/controller或已完成epoch。

|case|完整轮/400|接受更新|生产物理|采集/40|正式/200|
|---|---:|---:|---:|---:|---:|
|L005_41|300|69300|122900|30|0|
|L005_42|0|0|0|0|0|
|L005_43|0|0|0|0|0|
|L020_41|0|0|0|0|0|
|L020_42|0|0|0|0|0|
|L020_43|0|0|0|0|0|

累计300/2400轮、69300/633000接受、122900/1171800生产、62/546隔离、30/246准入autograd、30/240采集、0/1200正式。方向诊断最多216autograd另记。

下一120分钟复查，首次核自己的保持阶段准入及吞吐，此后依据整seed或整组正式实际剩余时间调整，不固定高频，不按时钟推断完成。

先完整读PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json、FINAL_WRAPPER_REVIEW.json及最新EVIDENCE/LOCAL_AUDIT。用户只授权OBJECT六run：0.05/0.2各41/42/43，已有0.1三个完整S只读复用。GOAL不跑，C/Q/旧机制不恢复。每阶段100轮，最新S双候选10步门控、零epsilon、自己成功缓存q和原4槽分母不变；只改变保持合成项λ_R，准入同λ均匀支持参照不是原0.1轨迹。

顺序L005三seed训练→600正式→L020三seed训练→600正式。240采集20–29不补采，1200正式0–19原600步5zero配对Normal。每stage自己的GPU准入/COPY_EQ继续；官方初态tensor/RNG仍须相同。用户授权3090硬件迁移，未来run Task6与原S全部tensor实际比较记录差异，不能预称相等；详见USER_RESUME_STATUS及RECOVERY_REPORT_20261102。未来阶段不能套PASS。

运输使用旧机制目录现有.session_control worker，先读status并核PID，不重启活跃worker或输出凭据。本工程engineering_resume_3090_20261102_v1/check复制旧目录queue，按result真实目录复制回来，audit_progress.py NEW PREVIOUS→record_progress.py NEW INTERVAL。任何错误先保留原失败、hash、进程/退出、完整恢复和已耗成本，禁止盲恢复部分块或环境attempt，不改科学协议。

只有自然engineering_resume_3090_20261102_v1/controller.exit0且无自身进程后，才final_readonly实际CPU24CP/6恢复/24teacher缓存/1200初态noise/54600门控及预算验收→audit_final_local→compare_primary_s。FINAL_WRAPPER_REVIEW目前仅STATIC_ONLY。包装错误保留原证据后外置版本化修复，不重训重评。

随后自主完成FINAL_REPORT/ACCEPTANCE：三档全结果、新旧任务、SF/FS、全部负单元、门控接受与梯度/训练代理、成本边界。不按正式成绩选择最佳λ，不扩seed或负结果加实验。已有0.1的600回合被两次配对而非1200独立基线回合。保留开发集反复使用和预训练来源未闭合，敏感性分析不是独立确认或全局无泄露。

每次检查按根AGENTS经publish.py轻量审查code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/model/state/RNG/cache/凭据/未审bundle。健康进展静默，只完成、新失败或需行动通知。六run全部独立验收、解释及发布后才删除本新监控，不恢复旧监控。
