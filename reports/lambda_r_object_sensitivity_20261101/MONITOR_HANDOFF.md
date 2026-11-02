## 最新2026-11-02 14:09 用户要求GPU利用率与全队列估时

check_readonly_20261102_140853审计PASS，132538全部前缀和135科学/5旧工程/8恢复工程项保持。L005_41 Task6/7/8各100轮、Task9完整39轮，共339轮83418接受151136完整轮生产、91自身隔离41准入autograd、30采集0正式；其余五run未开始。Task9自己的准入和BRANCH_COPY_EQ实际通过，controller2560/GPU2566健康。capacity_readonly_20261102_140929的30秒资源窗末状态已到340轮，完整逐轮审计仍以339快照为准。

30个1秒GPU采样均31.7%、0–43%，显存2872/24576MiB、56–62℃、功率均177.8W（上限350W），训练进程约99.5%单CPU核心。窗口末无活动温度/功率/硬件降频；GPU未持续满载，串行主机工作/同步/小算子可能相关，但这不是全run瓶颈定位或硬件无故障证明。0额外optimizer/环境/信号/生产写入。

恢复后Task8共40轮均53.714秒；Task9前39轮均65.210秒，近10轮65.521秒（60.22–69.13）。当前run余61轮纯66.61分钟，保存/最后10采集后规划75–95分钟。未来五run Task6/7未测，按单/双候选及批次比例外推；五run加当前余段条件纯训练25.05小时，计准备保存210剩余采集和资源/缓存波动，训练规划27–33小时，1200正式另2–4小时（历史600正式56.60分钟只作参考）。总训练+正式规划29–37小时，自14:09起约11月3日19:10至4日03:10；CPU独立验收/解释发布另计。不是置信区间或保证，.2组三seed和新3090正式尚未实测，不按时钟断言完成。详见GPU_TIMING_REPORT_20261102_1409.md及TIMING_AUDIT。下一75分钟在当前run结束较早侧复查，覆盖旧120分钟。

恢复与硬件授权保持：工程合同46bf5fe3bf0acfe9f23441a6c64c35d29438e8abdaef27262ffa5a809e6741e3；原260轮及4167门控保留，8137/47工程诊断和原中断140已记账候选+UNKNOWN单列。最终只认engineering_resume_3090_20261102_v1/controller.exit0及无自身进程，原根/旧工程exit不能伪造。未来stage自己的准入/BRANCH_COPY_EQ，未来run初始tensor/RNG同一及Task6跨GPU实际比较（不预称相等）仍须核验；硬件混杂/开发集/预训练来源限制进入最终解释。运输worker3700，先核status/PID，禁止重跑任何dispatch/controller/build/诊断或已完成轮。只用恢复工程check/audit/record/final包装。

# OBJECT 最新S固定λ_R六run：当前交接

实际观察 2026-11-02T14:08:53.496116+08:00；check_readonly_20261102_140853/LOCAL_AUDIT=PASS_PROGRESS_NOT_FINAL。

唯一root `/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1`，controller2560，合同`6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e`，135冻结项。唯一部署deploy_remote_20261101_152959。禁止重复compose/freeze/deploy/controller或已完成epoch。

|case|完整轮/400|接受更新|生产物理|采集/40|正式/200|
|---|---:|---:|---:|---:|---:|
|L005_41|339|83418|151136|30|0|
|L005_42|0|0|0|0|0|
|L005_43|0|0|0|0|0|
|L020_41|0|0|0|0|0|
|L020_42|0|0|0|0|0|
|L020_43|0|0|0|0|0|

累计339/2400轮、83418/633000接受、151136/1171800生产、91/546隔离、41/246准入autograd、30/240采集、0/1200正式。方向诊断最多216autograd另记。

下一75分钟复查，首次核自己的保持阶段准入及吞吐，此后依据整seed或整组正式实际剩余时间调整，不固定高频，不按时钟推断完成。

先完整读PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json、FINAL_WRAPPER_REVIEW.json及最新EVIDENCE/LOCAL_AUDIT。用户只授权OBJECT六run：0.05/0.2各41/42/43，已有0.1三个完整S只读复用。GOAL不跑，C/Q/旧机制不恢复。每阶段100轮，最新S双候选10步门控、零epsilon、自己成功缓存q和原4槽分母不变；只改变保持合成项λ_R，准入同λ均匀支持参照不是原0.1轨迹。

顺序L005三seed训练→600正式→L020三seed训练→600正式。240采集20–29不补采，1200正式0–19原600步5zero配对Normal。每stage自己的GPU准入/COPY_EQ继续；官方初态tensor/RNG仍须相同。用户授权3090硬件迁移，未来run Task6与原S全部tensor实际比较记录差异，不能预称相等；详见USER_RESUME_STATUS及RECOVERY_REPORT_20261102。未来阶段不能套PASS。

运输使用旧机制目录现有.session_control worker，先读status并核PID，不重启活跃worker或输出凭据。本工程engineering_resume_3090_20261102_v1/check复制旧目录queue，按result真实目录复制回来，audit_progress.py NEW PREVIOUS→record_progress.py NEW INTERVAL。任何错误先保留原失败、hash、进程/退出、完整恢复和已耗成本，禁止盲恢复部分块或环境attempt，不改科学协议。

只有自然engineering_resume_3090_20261102_v1/controller.exit0且无自身进程后，才final_readonly实际CPU24CP/6恢复/24teacher缓存/1200初态noise/54600门控及预算验收→audit_final_local→compare_primary_s。FINAL_WRAPPER_REVIEW目前仅STATIC_ONLY。包装错误保留原证据后外置版本化修复，不重训重评。

随后自主完成FINAL_REPORT/ACCEPTANCE：三档全结果、新旧任务、SF/FS、全部负单元、门控接受与梯度/训练代理、成本边界。不按正式成绩选择最佳λ，不扩seed或负结果加实验。已有0.1的600回合被两次配对而非1200独立基线回合。保留开发集反复使用和预训练来源未闭合，敏感性分析不是独立确认或全局无泄露。

每次检查按根AGENTS经publish.py轻量审查code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/model/state/RNG/cache/凭据/未审bundle。健康进展静默，只完成、新失败或需行动通知。六run全部独立验收、解释及发布后才删除本新监控，不恢复旧监控。
